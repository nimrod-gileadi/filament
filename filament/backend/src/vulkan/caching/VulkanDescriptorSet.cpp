/*
 * Copyright (C) 2024 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "VulkanDescriptorSet.h"
#include "utils/Panic.h"
#include "vulkan/VulkanHandles.h"
#include "vulkan/VulkanUtility.h"
#include "vulkan/vulkan_core.h"

#include <vulkan/VulkanConstants.h>
#include <vulkan/VulkanImageUtility.h>
#include <vulkan/VulkanResources.h>

#include <memory>
#include <type_traits>
#include <vector>

#include <math.h>

namespace filament::backend {

namespace {

using ImgUtil = VulkanImageUtility;
using Bitmask = VulkanDescriptorSetLayout::Bitmask;

constexpr uint8_t EXPECTED_IN_FLIGHT_FRAMES = 3;// Asssume triple buffering
constexpr uint8_t DESCRIPTOR_TYPE_COUNT = 3;

// This assumes we have at most 32-bound samplers, 10 UBOs and, 1 input attachment.
// TODO: Safe to remove after [UPCOMING_CHANGE]
constexpr uint8_t MAX_SAMPLER_BINDING = 32;
constexpr uint8_t MAX_UBO_BINDING = 10;
constexpr uint8_t MAX_INPUT_ATTACHMENT_BINDING = 1;
constexpr uint8_t MAX_BINDINGS = MAX_SAMPLER_BINDING + MAX_UBO_BINDING + MAX_INPUT_ATTACHMENT_BINDING;

// This struct can be used to indicate the count of each type of descriptor with respect to a
// layout, or it can be used to indicate the size and capacity of a descriptor pool.
struct DescriptorCount {
    uint32_t ubo = 0;
    uint32_t dynamicUbo = 0;
    uint32_t sampler = 0;
    uint32_t inputAttachment = 0;

    bool operator==(DescriptorCount const& right) const noexcept {
        return ubo == right.ubo && dynamicUbo == right.dynamicUbo &&
                sampler == right.sampler && inputAttachment == right.inputAttachment;
    }

    static inline DescriptorCount fromLayoutBitmask(Bitmask const& mask) {
        return {
            .ubo = BitCounter.count(collapse(mask.ubo)),
            .dynamicUbo = BitCounter.count(collapse(mask.dynamicUbo)),
            .sampler = BitCounter.count(collapse(mask.sampler)),
            .inputAttachment = BitCounter.count(collapse(mask.inputAttachment)),
        };
    }

    DescriptorCount operator*(uint16_t mult) const noexcept {
        // TODO: check for overflow.

        DescriptorCount ret;
        ret.ubo = ubo * mult;
        ret.dynamicUbo = dynamicUbo * mult;
        ret.sampler = sampler * mult;
        ret.inputAttachment = inputAttachment * mult;
        return ret;
    }
private:
    // We care about the number of bindings in both stages, so we collapse the mask into the lower
    // half (i.e. vertex stage) n-bits of the mask for counting the total number of unique
    // descriptors.
    template<typename MaskType>
    inline static MaskType collapse(MaskType mask) {
        constexpr uint8_t NBITS_DIV_2 = sizeof(MaskType) * 4;
        // First zero out the top-half and then or the bottom-half against the original top-half.
        return ((mask << NBITS_DIV_2) >> NBITS_DIV_2) | (mask >> NBITS_DIV_2);
    }
};

// We create a pool for each layout as defined by the number of descriptors of each type. For
// example, a layout of
// 'A' =>
//   layout(binding = 0, set = 1) uniform {};
//   layout(binding = 1, set = 1) sampler1;
//   layout(binding = 2, set = 1) sampler2;
//
// would be equivalent to
// 'B' =>
//   layout(binding = 1, set = 2) uniform {};
//   layout(binding = 2, set = 2) sampler2;
//   layout(binding = 3, set = 2) sampler3;
//
// TODO: we might do better if we understand the types of unique layouts and can combine them in a
// single pool without too much waste.
class DescriptorPool {
private:
    using Count = DescriptorCount;
public:
    DescriptorPool(VkDevice device, VulkanResourceAllocator* allocator,
            VulkanDescriptorSetLayout* layout, uint16_t capacity)
        : mDevice(device),
          mAllocator(allocator),
          mVkLayout(layout->vklayout),
          mCount(DescriptorCount::fromLayoutBitmask(layout->bitmask)),
          mCapacity(capacity) {
        Count const actual = mCount * capacity;
        VkDescriptorPoolSize sizes[4] = {
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = actual.ubo,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = actual.dynamicUbo,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = actual.sampler,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                .descriptorCount = actual.inputAttachment,
            },
        };
        VkDescriptorPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = capacity,
            .poolSizeCount = 4,
            .pPoolSizes = sizes,
        };
        vkCreateDescriptorPool(mDevice, &info, VKALLOC, &mPool);
    }

    DescriptorPool(DescriptorPool const&) = delete;
    DescriptorPool& operator=(DescriptorPool const&) = delete;

    ~DescriptorPool() {
        vkDestroyDescriptorPool(mDevice, mPool, VKALLOC);
    }

    uint16_t const& capacity() {
        return mCapacity;
    }

    // A convenience method for checking if this pool can allocate sets for a given layout.
    inline bool canAllocate(VulkanDescriptorSetLayout* layout) {
        return DescriptorCount::fromLayoutBitmask(layout->bitmask) == mCount;
    }

    Handle<VulkanDescriptorSet> obtainSet() {
        if (!mUnused.empty()) {
            auto set = mUnused.back();
            mUnused.pop_back();
            mSize++;
            return set;
        }
        if (mSize + 1 > mCapacity) {
            // This is the equivalent of returning null.
            return Handle<VulkanDescriptorSet>();
        }

        // Creating a new set
        VkDescriptorSetLayout layouts[1] = {mVkLayout};
        VkDescriptorSetAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = mPool,
            .descriptorSetCount = 1,
            .pSetLayouts = layouts,
        };
        VkDescriptorSet vkSet;
        UTILS_UNUSED VkResult result = vkAllocateDescriptorSets(mDevice, &allocInfo, &vkSet);
        ASSERT_POSTCONDITION(result == VK_SUCCESS, "Failed to allocate descriptor set code=%d",
                result);
        mSize++;
        return createSet(vkSet);
    }

private:
    Handle<VulkanDescriptorSet> createSet(VkDescriptorSet vkSet) {
        return mAllocator->initHandle<VulkanDescriptorSet>(vkSet, [this, vkSet]() {
            // We are recycling - release the vk resource back into the pool. Note that the
            // vk handle has not changed, but we need to change the backend handle to allow
            // for proper refcounting of resources referenced in this set.
            mUnused.push_back(createSet(vkSet));
            mSize--;
        });
    }

    VkDevice mDevice;
    VulkanResourceAllocator* mAllocator;
    VkDescriptorPool mPool;
    VkDescriptorSetLayout mVkLayout;
    Count mCount;
    uint16_t const mCapacity;

    // This tracks that currently the number of in-use descriptor sets.
    uint16_t mSize;

    // This maps a layout ot a list of descriptor sets allocated for that layout.
    std::vector<Handle<VulkanDescriptorSet>> mUnused;
};

class DescriptorInfinitePool {
private:
    static constexpr uint16_t EXPECTED_SET_COUNT = 10;
    static constexpr float SET_COUNT_GROWTH_FACTOR = 1.5;
public:
    DescriptorInfinitePool(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mResourceAllocator(allocator) {}

    Handle<VulkanDescriptorSet> obtainSet(VulkanDescriptorSetLayout* layout) {
        DescriptorPool* sameTypePool = nullptr;
        for (auto& pool: mPools) {
            if (!pool->canAllocate(layout)) {
                continue;
            }
            if (auto set = pool->obtainSet(); set) {
                return set;
            }
            if (!sameTypePool || sameTypePool->capacity() < pool->capacity()) {
                sameTypePool = pool.get();
            }
        }

        uint16_t capacity = EXPECTED_SET_COUNT;
        if (sameTypePool) {
            // Exponentially increase the size of the pool  to ensure we don't hit this too often.
            capacity = std::ceil(sameTypePool->capacity() * SET_COUNT_GROWTH_FACTOR);
        }

        // We need to increase the set of pools by one.
        mPools.push_back(
                std::make_unique<DescriptorPool>(mDevice, mResourceAllocator, layout, capacity));
        auto& pool = mPools.back();
        return pool->obtainSet();
    }

private:
    VkDevice mDevice;
    VulkanResourceAllocator* mResourceAllocator;
    std::vector<std::unique_ptr<DescriptorPool>> mPools;
};

class LayoutCache {
private:
    using Key = VulkanDescriptorSetLayout::Bitmask;

    // Make sure the key is 8-bytes aligned.
    static_assert(sizeof(Key) % 8 == 0);

    struct Equal {
        bool operator()(Key const& k1, Key const& k2) const {
            return 0 == memcmp((const void*) &k1, (const void*) &k2, sizeof(Key));
        }
    };

    using HashFn = utils::hash::MurmurHashFn<Key>;
    using LayoutMap = std::unordered_map<Key, Handle<VulkanDescriptorSetLayout>, HashFn, Equal>;

public:
    explicit LayoutCache(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mAllocator(allocator) {}

    ~LayoutCache() {
        for (auto [key, layout]: mLayouts) {
            auto layoutPtr = mAllocator->handle_cast<VulkanDescriptorSetLayout*>(layout);
            vkDestroyDescriptorSetLayout(mDevice, layoutPtr->vklayout, VKALLOC);
        }
        mLayouts.clear();
    }

    void destroyLayout(Handle<VulkanDescriptorSetLayout> handle) {
        auto layoutPtr = mAllocator->handle_cast<VulkanDescriptorSetLayout*>(handle);
        for (auto [key, layout]: mLayouts) {
            if (layout == handle) {
                mLayouts.erase(key);
                break;
            }
        }
        vkDestroyDescriptorSetLayout(mDevice, layoutPtr->vklayout, VKALLOC);
    }

    Handle<VulkanDescriptorSetLayout> getLayout(descset::DescriptorSetLayout const& layout) {
        Key key = Key::fromBackendLayout(layout);
        if (auto iter = mLayouts.find(key); iter != mLayouts.end()) {
            return iter->second;
        }

        VkDescriptorSetLayoutBinding toBind[MAX_BINDINGS];
        uint32_t count = 0;
        for (auto const& binding: layout.bindings) {
            auto& bindInfo = toBind[count];
            bindInfo.binding = binding.binding;
            bindInfo.descriptorCount = 1;
            auto& stages = bindInfo.stageFlags;
            auto& type = bindInfo.descriptorType;

            if (binding.stageFlags & descset::ShaderStageFlags2::VERTEX) {
                stages |= VK_SHADER_STAGE_VERTEX_BIT;
            }
            if (binding.stageFlags & descset::ShaderStageFlags2::FRAGMENT) {
                stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            switch (binding.type) {
                case descset::DescriptorType::UNIFORM_BUFFER: {
                    type = binding.flags == descset::DescriptorFlags::DYNAMIC_OFFSET ?
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC :
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    break;
                }
                case descset::DescriptorType::SAMPLER: {
                    type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    break;
                }
                case descset::DescriptorType::INPUT_ATTACHMENT: {
                    type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                    break;
                }
            }
            count++;
        }

        VkDescriptorSetLayoutCreateInfo dlinfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .bindingCount = count,
            .pBindings = toBind,
        };

        VkDescriptorSetLayout outLayout;
        vkCreateDescriptorSetLayout(mDevice, &dlinfo, VKALLOC, &outLayout);
        auto handle = mLayouts[key] = mAllocator->initHandle<VulkanDescriptorSetLayout>(outLayout);
        return handle;
    }

private:
    VkDevice mDevice;
    VulkanResourceAllocator* mAllocator;
    LayoutMap mLayouts;
};

}// anonymous namespace


class VulkanDescriptorSetManager::Impl {
private:
    using UBOMap = std::unordered_map<uint8_t, std::pair<VkDescriptorBufferInfo, VulkanBufferObject*>>;
    using SamplerMap = std::unordered_map<uint8_t, std::pair<VkDescriptorImageInfo, VulkanTexture*>>;
    using VulkanDescriptorSetLayoutList = VulkanDescriptorSetManager::VulkanDescriptorSetLayoutList;
    using GetPipelineLayoutFunction = VulkanDescriptorSetManager::GetPipelineLayoutFunction;

    struct BoundState {
        VkCommandBuffer cmdbuf = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VulkanDescriptorSet* set = nullptr;
        VulkanDescriptorSetLayoutList layouts;

        inline bool operator==(BoundState const& b) const {
            return set == b.set && cmdbuf == b.cmdbuf && pipelineLayout == b.pipelineLayout;
        }

        inline bool operator!=(BoundState const& b) const { return !(*this == b); }

        inline bool valid() {
            return cmdbuf != VK_NULL_HANDLE;
        }
    };
    
    static constexpr uint8_t UBO_SET_ID = 0;
    static constexpr uint8_t SAMPLER_SET_ID = 1;
    static constexpr uint8_t INPUT_ATTACHMENT_SET_ID = 2;
    
public:
    Impl(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mAllocator(allocator),
          mLayoutCache(device, allocator),
          mDescriptorPool(device, allocator),
          mHaveDynamicUbos(false),
          mResources(allocator) {}

    // This will write/update/bind all of the descriptor set.
    // When bind() is called, that's when the descriptor sets are allocated and then updated. This
    // behavior will change after the [UPCOMING CHANGE] completes.
    VkPipelineLayout bind(VulkanCommandBuffer* commands,
            VulkanDescriptorSetLayoutList const& layouts,
            GetPipelineLayoutFunction& getPipelineLayoutFn) {
        using DescriptorSetHandles =
                std::array<VkDescriptorSet, VulkanDescriptorSetManager::UNIQUE_DESCRIPOR_SET_COUNT>;

        DescriptorSetHandles descSets {VK_NULL_HANDLE};
        uint8_t descSetCount = 0;

        VkWriteDescriptorSet descriptorWrites[MAX_BINDINGS];
        uint32_t nwrites = 0;

        for (uint8_t i = 0; i < UNIQUE_DESCRIPOR_SET_COUNT; ++i) {
            auto handle = layouts[i];
            if (!handle) {
                continue;
            }
            switch (i) {
                case UBO_SET_ID:
                    break;                    
                case SAMPLER_SET_ID:
                    break;
                case INPUT_ATTACHMENT_SET_ID:
                    break;
            }
            auto layout = mAllocator->handle_cast<VulkanDescriptorSetLayout*>(handle);
            
        }
        
        auto const& uboMask = layout.ubo;
        auto const& samplerMask = layout.sampler;


        if (uboMask) {
            auto [descriptorSet, vkset, vklayout, writes] =
                writeUbos(uboMask, descriptorWrites, uboWrite, nwrites);

            nwrites = writes;
            descSets[descSetCount++] = vkset;
            commands->acquire(descriptorSet);
        }


        if (nwrites) {
            vkUpdateDescriptorSets(mDevice, nwrites, descriptorWrites, 0, nullptr);
        }

        VkPipelineLayout const pipelineLayout = getPipelineLayoutFn(layouts);


        VkCommandBuffer const cmdbuffer = commands->buffer();

        BoundState state {
            .cmdbuf = cmdbuffer,
            .pipelineLayout = pipelineLayout,
            .handles = descSets,
            .uboMask = uboMask,
        };

        if (state != mBoundState) {
            vkCmdBindDescriptorSets(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                    descSetCount, descSets.data(), 0, nullptr);
            mBoundState = state;
        }

        // Once bound, the resources are now ref'd in the descriptor set and the some resources can
        // be released and the descriptor set is ref'd by the command buffer.
        for (auto const& [binding, samplerBundle]: mSamplerMap) {
            auto const& [info, texture] = samplerBundle;
            mResources.release(texture);
        }
        mSamplerMap.clear();
        mInputAttachment = {};

        return pipelineLayout;
    }

    void dynamicBind(VulkanCommandBuffer* commands) {
        if (!mHaveDynamicUbos) {
            return;
        }
        assert_invariant(mBoundState.valid());

        VkWriteDescriptorSet descriptorWrites[MAX_UBO_BINDING];
        VkDescriptorBufferInfo uboWrite[MAX_UBO_BINDING];

        auto [descriptorSet, vkset, vklayout, nwrites] =
                writeUbos(mBoundState.uboMask, descriptorWrites, uboWrite, 0);

        if (nwrites == 0) {
            return;
        }

        commands->acquire(descriptorSet);

        vkUpdateDescriptorSets(mDevice, nwrites, descriptorWrites, 0, nullptr);

        // Only bind the UBO set
        vkCmdBindDescriptorSets(mBoundState.cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                mBoundState.pipelineLayout, 0, 1, &vkset, 0, nullptr);

        mHaveDynamicUbos = false;
    }

    Handle<VulkanDescriptorSetLayout> createLayout(descset::DescriptorSetLayout const& description) {
        return mLayoutCache.getLayout(description);
    }

    void destroyLayout(Handle<VulkanDescriptorSetLayout> layout) {
        mLayoutCache.destroyLayout(layout);
    }

    // Note that before [UPCOMING CHANGE] arrives, the "update" methods stash state within this
    // class and is not actually working with respect to a descriptor set.
    void updateBuffer(Handle<VulkanDescriptorSet>, uint8_t binding,
            VulkanBufferObject* bufferObject, VkDeviceSize offset, VkDeviceSize size) noexcept {
        VkDescriptorBufferInfo const info {
                .buffer = bufferObject->buffer.getGpuBuffer(),
                .offset = offset,
                .range = size,
        };
        mUboMap[binding] = {info, bufferObject};
        mResources.acquire(bufferObject);

        if (mHaveDynamicUbos && mBoundState.valid()) {
            mHaveDynamicUbos = true;
        }
    }

    void updateSampler(Handle<VulkanDescriptorSet>, uint8_t binding,
            VulkanTexture* texture, VkSampler sampler) noexcept {
        VkDescriptorImageInfo info {
                .sampler = sampler,
        };
        VkImageSubresourceRange const range = texture->getPrimaryViewRange();
        VkImageViewType const expectedType = texture->getViewType();
        if (any(texture->usage & TextureUsage::DEPTH_ATTACHMENT) &&
                expectedType == VK_IMAGE_VIEW_TYPE_2D) {
            // If the sampler is part of a mipmapped depth texture, where one of the level *can* be
            // an attachment, then the sampler for this texture has the same view properties as a
            // view for an attachment. Therefore, we can use getAttachmentView to get a
            // corresponding VkImageView.
            info.imageView = texture->getAttachmentView(range);
        } else {
            info.imageView = texture->getViewForType(range, expectedType);
        }
        info.imageLayout = ImgUtil::getVkLayout(texture->getPrimaryImageLayout());
        mSamplerMap[binding] = {info, texture};
        mResources.acquire(texture);
    }

    void updateInputAttachment(Handle<VulkanDescriptorSet>,
            VulkanAttachment attachment) noexcept {
        mInputAttachment = attachment;
        mResources.acquire(mInputAttachment.texture);
    }

    void clearBuffer(uint32_t binding) {
        if (auto itr = mUboMap.find(binding); itr != mUboMap.end()) {
            auto& [info, uboPtr] = itr->second;
            mResources.release(uboPtr);
            mUboMap.erase(itr);
        }
    }

    void setPlaceHolders(VkSampler sampler, VulkanTexture* texture,
            VulkanBufferObject* bufferObject) noexcept {
        mPlaceHolderSampler = sampler;
        mPlaceHolderTexture = texture;
        mPlaceHolderObject = bufferObject;
    }

    void clearState() noexcept {
        mHaveDynamicUbos = false;
        mInputAttachment = {};
        mResources.release(mInputAttachment.texture);
    }

private:

    void writeUbos(UniformBufferBitmask const& uboMask, VkWriteDescriptorSet* descriptorWrites,
            VkDescriptorBufferInfo* uboWrite, uint32_t nwrites) {
    }

    VkDevice mDevice;
    VulkanResourceAllocator* mAllocator;
    LayoutCache mLayoutCache;
    DescriptorInfinitePool mDescriptorPool;
    bool mHaveDynamicUbos;

    UBOMap mUboMap;
    SamplerMap mSamplerMap;
    VulkanAttachment mInputAttachment;

    VulkanResourceManager mResources;

    VkSampler mPlaceHolderSampler;
    VulkanTexture* mPlaceHolderTexture;
    VulkanBufferObject* mPlaceHolderObject;

    BoundState mBoundState;
};

VulkanDescriptorSetManager::VulkanDescriptorSetManager(VkDevice device,
        VulkanResourceAllocator* resourceAllocator)
    : mImpl(new Impl(device, resourceAllocator)) {}

void VulkanDescriptorSetManager::terminate() noexcept {
    assert_invariant(mImpl);
    delete mImpl;
    mImpl = nullptr;
}

// This will write/update/bind all of the descriptor set.
VkPipelineLayout VulkanDescriptorSetManager::bind(VulkanCommandBuffer* commands,
        VulkanDescriptorSetManager::VulkanDescriptorSetLayoutList const& layouts,
        VulkanDescriptorSetManager::GetPipelineLayoutFunction& getPipelineLayoutFn) {
    return mImpl->bind(commands, layouts, getPipelineLayoutFn);
}

void VulkanDescriptorSetManager::dynamicBind(VulkanCommandBuffer* commands, Handle<VulkanDescriptorSetLayout> uboLayout) {
    mImpl->dynamicBind(commands);
}

Handle<VulkanDescriptorSetLayout> VulkanDescriptorSetManager::createLayout(
        descset::DescriptorSetLayout const& layout) {
    return mImpl->createLayout(layout);
}

void VulkanDescriptorSetManager::destroyLayout(Handle<VulkanDescriptorSetLayout> layout) {
    mImpl->destroyLayout(layout);
}

void VulkanDescriptorSetManager::updateBuffer(Handle<VulkanDescriptorSet> set,
        uint8_t binding, VulkanBufferObject* bufferObject, VkDeviceSize offset,
        VkDeviceSize size) noexcept {
    mImpl->updateBuffer(set, binding, bufferObject, offset, size);
}

void VulkanDescriptorSetManager::updateSampler(Handle<VulkanDescriptorSet> set,
        uint8_t binding, VulkanTexture* texture, VkSampler sampler) noexcept {
    mImpl->updateSampler(set, binding, texture, sampler);
}

void VulkanDescriptorSetManager::updateInputAttachment(Handle<VulkanDescriptorSet> set, VulkanAttachment attachment) noexcept {
    mImpl->updateInputAttachment(set, attachment);
}

void VulkanDescriptorSetManager::clearBuffer(uint32_t bindingIndex) {
    mImpl->clearBuffer(bindingIndex);
}

void VulkanDescriptorSetManager::setPlaceHolders(VkSampler sampler, VulkanTexture* texture,
        VulkanBufferObject* bufferObject) noexcept {
    mImpl->setPlaceHolders(sampler, texture, bufferObject);
}

void VulkanDescriptorSetManager::clearState() noexcept { mImpl->clearState(); }


}// namespace filament::backend
