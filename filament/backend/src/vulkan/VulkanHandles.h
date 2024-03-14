/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef TNT_FILAMENT_BACKEND_VULKANHANDLES_H
#define TNT_FILAMENT_BACKEND_VULKANHANDLES_H

// This needs to be at the top
#include "DriverBase.h"

#include "VulkanBuffer.h"
#include "VulkanResources.h"
#include "VulkanSwapChain.h"
#include "VulkanTexture.h"
#include "VulkanUtility.h"

#include "private/backend/SamplerGroup.h"
#include "vulkan/vulkan_core.h"

#include <utils/Mutex.h>
#include <utils/StructureOfArrays.h>

namespace filament::backend {

using namespace descset;

class VulkanTimestamps;

struct VulkanDescriptorSetLayout : public VulkanResource {     
    static constexpr uint8_t UNIQUE_DESCRIPOR_SET_COUNT = 3;
     
    // The bitmask representation of a set layout.
    struct Bitmask {
        UniformBufferBitmask ubo = 0;
        UniformBufferBitmask dynamicUbo = 0;
        SamplerBitmask sampler = 0;
        InputAttachmentBitmask inputAttachment = 0;

        static Bitmask fromBackendLayout(descset::DescriptorSetLayout const& layout);
    };
    
    explicit VulkanDescriptorSetLayout(VkDescriptorSetLayout layout, Bitmask const& bitmask)
        : VulkanResource(VulkanResourceType::DESCRIPTOR_SET_LAYOUT),
          vklayout(layout),
          bitmask(bitmask) {}

    VkDescriptorSetLayout const vklayout;
    Bitmask const bitmask;
};

struct VulkanProgram : public HwProgram, VulkanResource {

    VulkanProgram(VkDevice device, Program const& builder) noexcept;

    using BindingList = CappedArray<uint16_t, MAX_SAMPLER_COUNT>;

    ~VulkanProgram();

    inline VkShaderModule getVertexShader() const {
        return mInfo->shaders[0];
    }

    inline VkShaderModule getFragmentShader() const { return mInfo->shaders[1]; }

    inline utils::FixedCapacityVector<uint16_t> const& getBindingToSamplerIndex() const {
        return mInfo->bindingToSamplerIndex;
    }

    inline BindingList const& getBindings() const {
        return mInfo->bindings;
    }

    // TODO: this is currently not used.
    // inline descset::DescriptorSetLayout const& getBindingLayout() const {
    //    return mInfo->layout;
    //}

    // TODO: in the usual case, we would have just one layout per program. But in the current setup,
    // we have a set/layout for each descriptor type. This will be changed in the future.
    using LayoutDescriptionList = std::array<descset::DescriptorSetLayout,
            VulkanDescriptorSetLayout::UNIQUE_DESCRIPOR_SET_COUNT>;
    inline LayoutDescriptionList const& getLayoutDescriptionList() const { return mInfo->layouts; }

#if FVK_ENABLED_DEBUG_SAMPLER_NAME
    inline utils::FixedCapacityVector<std::string> const& getBindingToName() const {
        return mInfo->bindingToName;
    }
#endif

private:
    // TODO: handle compute shaders.
    // The expected order of shaders - from frontend to backend - is vertex, fragment, compute.
    static constexpr uint8_t const MAX_SHADER_MODULES = 2;

    struct PipelineInfo {
        PipelineInfo()
            : bindingToSamplerIndex(MAX_SAMPLER_COUNT, 0xffff)
#if FVK_ENABLED_DEBUG_SAMPLER_NAME
            , bindingToName(MAX_SAMPLER_COUNT, "")
#endif
            {}

        // This bitset maps to each of the sampler in the sampler groups associated with this
        // program, and whether each sampler is used in which shader (i.e. vert, frag, compute).
        UsageFlags usage;

        BindingList bindings;

        // We store the samplerGroupIndex as the top 8-bit and the index within each group as the lower 8-bit.
        utils::FixedCapacityVector<uint16_t> bindingToSamplerIndex;
        VkShaderModule shaders[MAX_SHADER_MODULES] = { VK_NULL_HANDLE };

        LayoutDescriptionList layouts;
        
        // TODO: this is currently not used.        
        //descset::DescriptorSetLayout layout;

#if FVK_ENABLED_DEBUG_SAMPLER_NAME
        // We store the sampler name mapped from binding index (only for debug purposes).
        utils::FixedCapacityVector<std::string> bindingToName;
#endif

    };

    PipelineInfo* mInfo;
    VkDevice mDevice = VK_NULL_HANDLE;
};

// The render target bundles together a set of attachments, each of which can have one of the
// following ownership semantics:
//
// - The attachment's VkImage is shared and the owner is VulkanSwapChain (mOffscreen = false).
// - The attachment's VkImage is shared and the owner is VulkanTexture   (mOffscreen = true).
//
// We use private inheritance to shield clients from the width / height fields in HwRenderTarget,
// which are not representative when this is the default render target.
struct VulkanRenderTarget : private HwRenderTarget, VulkanResource {
    // Creates an offscreen render target.
    VulkanRenderTarget(VkDevice device, VkPhysicalDevice physicalDevice,
            VulkanContext const& context, VmaAllocator allocator,
            VulkanCommands* commands, uint32_t width, uint32_t height,
            uint8_t samples, VulkanAttachment color[MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT],
            VulkanAttachment depthStencil[2], VulkanStagePool& stagePool);

    // Creates a special "default" render target (i.e. associated with the swap chain)
    explicit VulkanRenderTarget();

    void transformClientRectToPlatform(VkRect2D* bounds) const;
    void transformClientRectToPlatform(VkViewport* bounds) const;
    VkExtent2D getExtent() const;
    VulkanAttachment getColor(int target) const;
    VulkanAttachment getMsaaColor(int target) const;
    VulkanAttachment getDepth() const;
    VulkanAttachment getMsaaDepth() const;
    uint8_t getColorTargetCount(const VulkanRenderPass& pass) const;
    uint8_t getSamples() const { return mSamples; }
    bool hasDepth() const { return mDepth.texture; }
    bool isSwapChain() const { return !mOffscreen; }
    void bindToSwapChain(VulkanSwapChain& surf);

private:
    VulkanAttachment mColor[MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT] = {};
    VulkanAttachment mDepth = {};
    VulkanAttachment mMsaaAttachments[MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT] = {};
    VulkanAttachment mMsaaDepthAttachment = {};
    const bool mOffscreen : 1;
    uint8_t mSamples : 7;
};

struct VulkanBufferObject;

struct VulkanVertexBufferInfo : public HwVertexBufferInfo, VulkanResource {
    VulkanVertexBufferInfo(uint8_t bufferCount, uint8_t attributeCount,
            AttributeArray const& attributes);

    inline VkVertexInputAttributeDescription const* getAttribDescriptions() const {
        return mInfo.mSoa.data<PipelineInfo::ATTRIBUTE_DESCRIPTION>();
    }

    inline VkVertexInputBindingDescription const* getBufferDescriptions() const {
        return mInfo.mSoa.data<PipelineInfo::BUFFER_DESCRIPTION>();
    }

    inline int8_t const* getAttributeToBuffer() const {
        return mInfo.mSoa.data<PipelineInfo::ATTRIBUTE_TO_BUFFER_INDEX>();
    }

    inline VkDeviceSize const* getOffsets() const {
        return mInfo.mSoa.data<PipelineInfo::OFFSETS>();
    }

    size_t getAttributeCount() const noexcept {
        return mInfo.mSoa.size();
    }

private:
    struct PipelineInfo {
        PipelineInfo(size_t size) : mSoa(size /* capacity */) {
            mSoa.resize(size);
        }

        // These correspond to the index of the element in the SoA
        static constexpr uint8_t ATTRIBUTE_DESCRIPTION = 0;
        static constexpr uint8_t BUFFER_DESCRIPTION = 1;
        static constexpr uint8_t OFFSETS = 2;
        static constexpr uint8_t ATTRIBUTE_TO_BUFFER_INDEX = 3;

        utils::StructureOfArrays<
            VkVertexInputAttributeDescription,
            VkVertexInputBindingDescription,
            VkDeviceSize,
            int8_t
        > mSoa;
    };

    PipelineInfo mInfo;
};

struct VulkanVertexBuffer : public HwVertexBuffer, VulkanResource {
    VulkanVertexBuffer(VulkanContext& context, VulkanStagePool& stagePool,
            VulkanResourceAllocator* allocator,
            uint32_t vertexCount, Handle<HwVertexBufferInfo> vbih);

    void setBuffer(VulkanResourceAllocator const& allocator,
            VulkanBufferObject* bufferObject, uint32_t index);

    inline VkBuffer const* getVkBuffers() const {
        return mBuffers.data();
    }

    inline VkBuffer* getVkBuffers() {
        return mBuffers.data();
    }

    Handle<HwVertexBufferInfo> vbih;
private:
    utils::FixedCapacityVector<VkBuffer> mBuffers;
    FixedSizeVulkanResourceManager<MAX_VERTEX_BUFFER_COUNT> mResources;
};

struct VulkanIndexBuffer : public HwIndexBuffer, VulkanResource {
    VulkanIndexBuffer(VmaAllocator allocator, VulkanStagePool& stagePool, uint8_t elementSize,
            uint32_t indexCount)
        : HwIndexBuffer(elementSize, indexCount),
          VulkanResource(VulkanResourceType::INDEX_BUFFER),
          buffer(allocator, stagePool, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, elementSize * indexCount),
          indexType(elementSize == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32) {}

    VulkanBuffer buffer;
    const VkIndexType indexType;
};

struct VulkanBufferObject : public HwBufferObject, VulkanResource {
    VulkanBufferObject(VmaAllocator allocator, VulkanStagePool& stagePool, uint32_t byteCount,
            BufferObjectBinding bindingType);

    VulkanBuffer buffer;
    const BufferObjectBinding bindingType;
};

struct VulkanSamplerGroup : public HwSamplerGroup, VulkanResource {
    // NOTE: we have to use out-of-line allocation here because the size of a Handle<> is limited
    std::unique_ptr<SamplerGroup> sb;// FIXME: this shouldn't depend on filament::SamplerGroup
    explicit VulkanSamplerGroup(size_t size) noexcept
        : VulkanResource(VulkanResourceType::SAMPLER_GROUP),
          sb(new SamplerGroup(size)) {}
};

struct VulkanRenderPrimitive : public HwRenderPrimitive, VulkanResource {
    VulkanRenderPrimitive(VulkanResourceAllocator* resourceAllocator,
            PrimitiveType pt, Handle<HwVertexBuffer> vbh, Handle<HwIndexBuffer> ibh);

    ~VulkanRenderPrimitive() {
        mResources.clear();
    }

    VulkanVertexBuffer* vertexBuffer = nullptr;
    VulkanIndexBuffer* indexBuffer = nullptr;

private:
    // Keep references to the vertex buffer and the index buffer.
    FixedSizeVulkanResourceManager<2> mResources;
};

struct VulkanFence : public HwFence, VulkanResource {
    VulkanFence()
        : VulkanResource(VulkanResourceType::FENCE) {}

    explicit VulkanFence(std::shared_ptr<VulkanCmdFence> fence)
        : VulkanResource(VulkanResourceType::FENCE),
          fence(fence) {}

    std::shared_ptr<VulkanCmdFence> fence;
};

struct VulkanTimerQuery : public HwTimerQuery, VulkanThreadSafeResource {
    explicit VulkanTimerQuery(std::tuple<uint32_t, uint32_t> indices);
    ~VulkanTimerQuery();

    void setFence(std::shared_ptr<VulkanCmdFence> fence) noexcept;

    bool isCompleted() noexcept;

    uint32_t getStartingQueryIndex() const {
        return mStartingQueryIndex;
    }

    uint32_t getStoppingQueryIndex() const {
        return mStoppingQueryIndex;
    }

private:
    uint32_t mStartingQueryIndex;
    uint32_t mStoppingQueryIndex;

    std::shared_ptr<VulkanCmdFence> mFence;
    utils::Mutex mFenceMutex;
};

struct VulkanDescriptorSet : public VulkanResource {
public:
    // Because we need to recycle descriptor set not used, we allow for a callback that the "Pool"
    // can use to repackage the vk handle.
    using OnRecycle = std::function<void(VulkanDescriptorSet*)>;

    VulkanDescriptorSet(VulkanResourceAllocator* allocator,
            VkDescriptorSet rawSet, OnRecycle&& onRecycleFn)
        : VulkanResource(VulkanResourceType::DESCRIPTOR_SET),
          resources(allocator),
          vkSet(rawSet),
          mOnRecycleFn(std::move(onRecycleFn)) {}

    ~VulkanDescriptorSet() {
        if (mOnRecycleFn) {
            mOnRecycleFn(this);
        }
    }
    
    // TODO: maybe change to fixed size for performance.
    VulkanAcquireOnlyResourceManager resources;
    VkDescriptorSet const vkSet;

private:
    OnRecycle mOnRecycleFn;
};

inline constexpr VkBufferUsageFlagBits getBufferObjectUsage(
        BufferObjectBinding bindingType) noexcept {
    switch(bindingType) {
        case BufferObjectBinding::VERTEX:
            return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        case BufferObjectBinding::UNIFORM:
            return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        case BufferObjectBinding::SHADER_STORAGE:
            return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        // when adding more buffer-types here, make sure to update VulkanBuffer::loadFromCpu()
        // if necessary.
    }
}

} // namespace filament::backend

#endif // TNT_FILAMENT_BACKEND_VULKANHANDLES_H
