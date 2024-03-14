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

#include "VulkanHandles.h"

#include "VulkanDriver.h"
#include "VulkanConstants.h"
#include "VulkanDriver.h"
#include "VulkanMemory.h"
#include "VulkanUtility.h"
#include "spirv/VulkanSpirvUtils.h"

#include <backend/platforms/VulkanPlatform.h>

#include <utils/Panic.h>    // ASSERT_POSTCONDITION

using namespace bluevk;

namespace filament::backend {

namespace {

void flipVertically(VkRect2D* rect, uint32_t framebufferHeight) {
    rect->offset.y = framebufferHeight - rect->offset.y - rect->extent.height;
}

void flipVertically(VkViewport* rect, uint32_t framebufferHeight) {
    rect->y = framebufferHeight - rect->y - rect->height;
}

void clampToFramebuffer(VkRect2D* rect, uint32_t fbWidth, uint32_t fbHeight) {
    int32_t x = std::max(rect->offset.x, 0);
    int32_t y = std::max(rect->offset.y, 0);
    int32_t right = std::min(rect->offset.x + (int32_t) rect->extent.width, (int32_t) fbWidth);
    int32_t top = std::min(rect->offset.y + (int32_t) rect->extent.height, (int32_t) fbHeight);
    rect->offset.x = std::min(x, (int32_t) fbWidth);
    rect->offset.y = std::min(y, (int32_t) fbHeight);
    rect->extent.width = std::max(right - x, 0);
    rect->extent.height = std::max(top - y, 0);
}

template<typename Bitmask>
static constexpr Bitmask fromStageFlags(ShaderStageFlags2 flags, uint8_t binding) {
    Bitmask ret;
    if (flags & ShaderStageFlags2::VERTEX) {
        ret |= (getVertexStage<Bitmask>() << binding);
    }
    if (flags & ShaderStageFlags2::FRAGMENT) {
        ret |= (getFragmentStage<Bitmask>() << binding);
    }
    return ret;
}


template <typename T>
std::string printx(T x) {
    std::string o = "0x";
    for (size_t i = 0; i < sizeof(x) * 8; ++i) {
        if (i%16 == 0 && i > 0) o+="-";
        if (x & (1 << i)) {
            o+="1";
        } else {
            o+="0";
        }
    }
    return o;
}

constexpr uint8_t UBO_MODULE_OFFSET = (sizeof(UniformBufferBitmask) * 8) / MAX_SHADER_MODULES;
constexpr uint8_t SAMPLER_MODULE_OFFSET = (sizeof(SamplerBitmask) * 8) / MAX_SHADER_MODULES;
constexpr uint8_t INPUT_ATTACHMENT_MODULE_OFFSET =
        (sizeof(InputAttachmentBitmask) * 8) / MAX_SHADER_MODULES;

template<typename Bitmask>
void addDescriptors(Bitmask mask, LayoutDescriptionList& outputList) {
    constexpr uint8_t MODULE_OFFSET + (sizeof(Bitmask) * 8) / MAX_SHADER_MODULES;
    for (uint8_t i = 0; i < sizeof(Bitmask) / MAX_SHADER_MODULES; ++i) {
        bool hasVertex = (mask & i) != 0;
        bool hasFragment = (mask & (1 << MODULE_OFFSET)) != 0;
        if (!hasVertex && !hasFragment) {
            continue;
        }

        DescriptorSetLayoutBinding binding {
            .stageFlags = (hasVertex ? ShaderStageFlags2::VERTEX : 0) |
                          (hasFragment ? ShaderStageFlags2::Fragment : 0);
            .binding = i, .flags = DescriptorFlags::NONE,
            .count = 0,// This is always 0 for now as we pass the size of the UBOs in the Driver API
                       // instead.
        };

        if constexpr (std::is_same_v<Bitmask, UniformBufferBitmask>) {
            binding.type = DescriptorType::UNIFORM_BUFFER;
        } else if constexpr (std::is_same_v<Bitmask, SamplerBitmask>) {
            binding.type = DescriptorType::SAMPLER;
        } else if constexpr (std::is_same_v<Bitmask, InputAttachmentBitmask>) {
            binding.type = DescriptorType::INPUT_ATTACHMENT;
        }
        outputList.push_back(binding);
    }

} // anonymous namespace

VulkanProgram::VulkanProgram(VkDevice device, Program const& builder) noexcept
    : HwProgram(builder.getName()),
      VulkanResource(VulkanResourceType::PROGRAM),
      mInfo(new PipelineInfo()),
      mDevice(device) {
    Program::ShaderSource const& blobs = builder.getShadersSource();
    auto& modules = mInfo->shaders;

    auto const& specializationConstants = builder.getSpecializationConstants();

    std::vector<uint32_t> shader;

    // TODO: this will be moved out of the shader as the descriptor set layout will be provided by
    // Filament instead of parsed from the shaders.
    UniformBufferBitmask uboMask;
    SamplerBitmask samplerMask;
    InputAttachmentBitmask inputAttachmentMask;

    static_assert(static_cast<ShaderStage>(0) == ShaderStage::VERTEX &&
            static_cast<ShaderStage>(1) == ShaderStage::FRAGMENT &&
            MAX_SHADER_MODULES ==2);

    // We use the counters to count the number of descriptors-per-type for sizing the Layout
    // 'bindings'.
    uint32_t uboCounter = 0, samplerCounter = 0, inputAttachmentCounter = 0;

    for (size_t i = 0; i < MAX_SHADER_MODULES; i++) {
        Program::ShaderBlob const& blob = blobs[i];

        uint32_t* data = (uint32_t*) blob.data();
        size_t dataSize = blob.size();

        if (!specializationConstants.empty()) {
            workaroundSpecConstant(blob, specializationConstants, shader);
            data = (uint32_t*) shader.data();
            dataSize = shader.size() * 4;
        }

        auto const [ubo, sampler, inputAttachment] = getProgramBindings(blob);
        uboCounter |= ubo;
        samplerCounter |= sampler;
        inputAttachmentCounter |= inputAttachment;
        uboMask |= (static_cast<UniformBufferBitmask>(ubo) << (UBO_MODULE_OFFSET * i));
        samplerMask |= (static_cast<SamplerBitmask>(sampler) << (SAMPLER_MODULE_OFFSET * i));
        inputAttachmentMask |= (static_cast<InputAttachmentBitmask>(inputAttachment)
                                << (INPUT_ATTACHMENT_MODULE_OFFSET * i));

        VkShaderModule& module = modules[i];
        VkShaderModuleCreateInfo moduleInfo = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = dataSize,
            .pCode = data,
        };
        VkResult result = vkCreateShaderModule(mDevice, &moduleInfo, VKALLOC, &module);
        ASSERT_POSTCONDITION(result == VK_SUCCESS, "Unable to create shader module.");

#if FVK_ENABLED(FVK_DEBUG_DEBUG_UTILS)
        std::string name{ builder.getName().c_str(), builder.getName().size() };
        switch (static_cast<ShaderStage>(i)) {
            case ShaderStage::VERTEX:
                name += "_vs";
                break;
            case ShaderStage::FRAGMENT:
                name += "_fs";
                break;
            default:
                PANIC_POSTCONDITION("Unexpected stage");
                break;
        }
        VulkanDriver::DebugUtils::setName(VK_OBJECT_TYPE_SHADER_MODULE,
                reinterpret_cast<uint64_t>(module), name.c_str());
#endif
    }

    size_t const totalDescriptors = BitCounter.count(uboCounter) + BitCounter.count(samplerCount) +
                                    BitCounter.count(inputAttachmentCounter);

#if FVK_ENABLED_DEBUG_SAMPLER_NAME
    auto& bindingToName = mInfo->bindingToName;
#endif

    auto& groupInfo = builder.getSamplerGroupInfo();
    auto& bindingToSamplerIndex = mInfo->bindingToSamplerIndex;
    auto& bindings = mInfo->bindings;
    for (uint8_t groupInd = 0; groupInd < Program::SAMPLER_BINDING_COUNT; groupInd++) {
        auto const& group = groupInfo[groupInd];
        auto const& samplers = group.samplers;
        for (size_t i = 0; i < samplers.size(); ++i) {
            uint32_t const binding = samplers[i].binding;
            bindingToSamplerIndex[binding] = (groupInd << 8) | (0xff & i);
            assert_invariant(bindings.find(binding) == bindings.end());
            bindings.insert(binding);

#if FVK_ENABLED_DEBUG_SAMPLER_NAME
            bindingToName[binding] = samplers[i].name.c_str();
#endif
        }
    }

#if FVK_ENABLED(FVK_DEBUG_SHADER_MODULE)
    utils::slog.d << "Created VulkanProgram " << builder << ", shaders = (" << modules[0]
                  << ", " << modules[1] << ")" << utils::io::endl;
#endif
}

VulkanProgram::~VulkanProgram() {
    for (auto shader: mInfo->shaders) {
        vkDestroyShaderModule(mDevice, shader, VKALLOC);
    }
    delete mInfo;
}

// Creates a special "default" render target (i.e. associated with the swap chain)
VulkanRenderTarget::VulkanRenderTarget() :
    HwRenderTarget(0, 0),
    VulkanResource(VulkanResourceType::RENDER_TARGET),
    mOffscreen(false), mSamples(1) {}

void VulkanRenderTarget::bindToSwapChain(VulkanSwapChain& swapChain) {
    assert_invariant(!mOffscreen);
    VkExtent2D const extent = swapChain.getExtent();
    mColor[0] = { .texture = swapChain.getCurrentColor() };
    mDepth = { .texture = swapChain.getDepth() };
    width = extent.width;
    height = extent.height;
}

VulkanRenderTarget::VulkanRenderTarget(VkDevice device, VkPhysicalDevice physicalDevice,
        VulkanContext const& context, VmaAllocator allocator, VulkanCommands* commands,
        uint32_t width, uint32_t height, uint8_t samples,
        VulkanAttachment color[MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT],
        VulkanAttachment depthStencil[2], VulkanStagePool& stagePool)
    : HwRenderTarget(width, height),
      VulkanResource(VulkanResourceType::RENDER_TARGET),
      mOffscreen(true),
      mSamples(samples) {
    for (int index = 0; index < MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT; index++) {
        mColor[index] = color[index];
    }
    mDepth = depthStencil[0];
    VulkanTexture* depthTexture = (VulkanTexture*) mDepth.texture;

    if (samples == 1) {
        return;
    }

    // Constrain the sample count according to both kinds of sample count masks obtained from
    // VkPhysicalDeviceProperties. This is consistent with the VulkanTexture constructor.
    const auto& limits = context.getPhysicalDeviceLimits();
    mSamples = samples = reduceSampleCount(samples, limits.framebufferDepthSampleCounts &
            limits.framebufferColorSampleCounts);

    // Create sidecar MSAA textures for color attachments if they don't already exist.
    for (int index = 0; index < MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT; index++) {
        const VulkanAttachment& spec = color[index];
        VulkanTexture* texture = (VulkanTexture*) spec.texture;
        if (texture && texture->samples == 1) {
            auto msTexture = texture->getSidecar();
            if (UTILS_UNLIKELY(!msTexture)) {
                // TODO: This should be allocated with the ResourceAllocator.
                msTexture = new VulkanTexture(device, physicalDevice, context, allocator, commands,
                        texture->target, ((VulkanTexture const*) texture)->levels, texture->format,
                        samples, texture->width, texture->height, texture->depth, texture->usage,
                        stagePool, true /* heap allocated */);
                texture->setSidecar(msTexture);
            }
            mMsaaAttachments[index] = {.texture = msTexture};
        }
        if (texture && texture->samples > 1) {
            mMsaaAttachments[index] = mColor[index];
        }
    }

    if (!depthTexture) {
        return;
    }

    // There is no need for sidecar depth if the depth texture is already MSAA.
    if (depthTexture->samples > 1) {
        mMsaaDepthAttachment = mDepth;
        return;
    }

    // MSAA depth texture must have the mipmap count of 1
    uint8_t const msLevel = 1;

    // Create sidecar MSAA texture for the depth attachment if it does not already exist.
    VulkanTexture* msTexture = depthTexture->getSidecar();
    if (UTILS_UNLIKELY(!msTexture)) {
        msTexture = new VulkanTexture(device, physicalDevice, context, allocator,
                commands, depthTexture->target, msLevel, depthTexture->format, samples,
                depthTexture->width, depthTexture->height, depthTexture->depth, depthTexture->usage,
                stagePool, true /* heap allocated */);
        depthTexture->setSidecar(msTexture);
    }

    mMsaaDepthAttachment = {
        .texture = msTexture,
        .level = msLevel,
        .layer = mDepth.layer,
    };
}

void VulkanRenderTarget::transformClientRectToPlatform(VkRect2D* bounds) const {
    const auto& extent = getExtent();
    flipVertically(bounds, extent.height);
    clampToFramebuffer(bounds, extent.width, extent.height);
}

void VulkanRenderTarget::transformClientRectToPlatform(VkViewport* bounds) const {
    flipVertically(bounds, getExtent().height);
}

VkExtent2D VulkanRenderTarget::getExtent() const {
    return {width, height};
}

VulkanAttachment VulkanRenderTarget::getColor(int target) const {
    return mColor[target];
}

VulkanAttachment VulkanRenderTarget::getMsaaColor(int target) const {
    return mMsaaAttachments[target];
}

VulkanAttachment VulkanRenderTarget::getDepth() const {
    return mDepth;
}

VulkanAttachment VulkanRenderTarget::getMsaaDepth() const {
    return mMsaaDepthAttachment;
}

uint8_t VulkanRenderTarget::getColorTargetCount(const VulkanRenderPass& pass) const {
    if (!mOffscreen) {
        return 1;
    }
    uint8_t count = 0;
    for (uint8_t i = 0; i < MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT; i++) {
        if (!mColor[i].texture) {
            continue;
        }
        // NOTE: This must be consistent with VkRenderPass construction (see VulkanFboCache).
        if (!(pass.params.subpassMask & (1 << i)) || pass.currentSubpass == 1) {
            count++;
        }
    }
    return count;
}

VulkanVertexBufferInfo::VulkanVertexBufferInfo(
        uint8_t bufferCount, uint8_t attributeCount, AttributeArray const& attributes)
    : HwVertexBufferInfo(bufferCount, attributeCount),
      VulkanResource(VulkanResourceType::VERTEX_BUFFER_INFO),
      mInfo(attributes.size()) {

    auto attribDesc = mInfo.mSoa.data<PipelineInfo::ATTRIBUTE_DESCRIPTION>();
    auto bufferDesc = mInfo.mSoa.data<PipelineInfo::BUFFER_DESCRIPTION>();
    auto offsets = mInfo.mSoa.data<PipelineInfo::OFFSETS>();
    auto attribToBufferIndex = mInfo.mSoa.data<PipelineInfo::ATTRIBUTE_TO_BUFFER_INDEX>();
    std::fill(mInfo.mSoa.begin<PipelineInfo::ATTRIBUTE_TO_BUFFER_INDEX>(),
            mInfo.mSoa.end<PipelineInfo::ATTRIBUTE_TO_BUFFER_INDEX>(), -1);

    for (uint32_t attribIndex = 0; attribIndex < attributes.size(); attribIndex++) {
        Attribute attrib = attributes[attribIndex];
        bool const isInteger = attrib.flags & Attribute::FLAG_INTEGER_TARGET;
        bool const isNormalized = attrib.flags & Attribute::FLAG_NORMALIZED;
        VkFormat vkformat = getVkFormat(attrib.type, isNormalized, isInteger);

        // HACK: Re-use the positions buffer as a dummy buffer for disabled attributes. Filament's
        // vertex shaders declare all attributes as either vec4 or uvec4 (the latter for bone
        // indices), and positions are always at least 32 bits per element. Therefore we can assign
        // a dummy type of either R8G8B8A8_UINT or R8G8B8A8_SNORM, depending on whether the shader
        // expects to receive floats or ints.
        if (attrib.buffer == Attribute::BUFFER_UNUSED) {
            vkformat = isInteger ? VK_FORMAT_R8G8B8A8_UINT : VK_FORMAT_R8G8B8A8_SNORM;
            attrib = attributes[0];
        }
        offsets[attribIndex] = attrib.offset;
        attribDesc[attribIndex] = {
            .location = attribIndex,// matches the GLSL layout specifier
            .binding = attribIndex, // matches the position within vkCmdBindVertexBuffers
            .format = vkformat,
        };
        bufferDesc[attribIndex] = {
            .binding = attribIndex,
            .stride = attrib.stride,
        };
        attribToBufferIndex[attribIndex] = attrib.buffer;
    }
}

VulkanVertexBuffer::VulkanVertexBuffer(VulkanContext& context, VulkanStagePool& stagePool,
        VulkanResourceAllocator* allocator,
        uint32_t vertexCount, Handle<HwVertexBufferInfo> vbih)
    : HwVertexBuffer(vertexCount),
      VulkanResource(VulkanResourceType::VERTEX_BUFFER),
      vbih(vbih),
      mBuffers(MAX_VERTEX_BUFFER_COUNT), // TODO: can we do better here?
      mResources(allocator) {
}

void VulkanVertexBuffer::setBuffer(VulkanResourceAllocator const& allocator,
        VulkanBufferObject* bufferObject, uint32_t index) {
    VulkanVertexBufferInfo const* const vbi =
            const_cast<VulkanResourceAllocator&>(allocator).handle_cast<VulkanVertexBufferInfo*>(vbih);
    size_t const count = vbi->getAttributeCount();
    VkBuffer* const vkbuffers = getVkBuffers();
    int8_t const* const attribToBuffer = vbi->getAttributeToBuffer();
    for (uint8_t attribIndex = 0; attribIndex < count; attribIndex++) {
        if (attribToBuffer[attribIndex] == static_cast<int8_t>(index)) {
            vkbuffers[attribIndex] = bufferObject->buffer.getGpuBuffer();
        }
    }
    mResources.acquire(bufferObject);
}

VulkanBufferObject::VulkanBufferObject(VmaAllocator allocator, VulkanStagePool& stagePool,
        uint32_t byteCount, BufferObjectBinding bindingType)
    : HwBufferObject(byteCount),
      VulkanResource(VulkanResourceType::BUFFER_OBJECT),
      buffer(allocator, stagePool, getBufferObjectUsage(bindingType), byteCount),
      bindingType(bindingType) {}

VulkanTimerQuery::VulkanTimerQuery(std::tuple<uint32_t, uint32_t> indices)
    : VulkanThreadSafeResource(VulkanResourceType::TIMER_QUERY),
      mStartingQueryIndex(std::get<0>(indices)),
      mStoppingQueryIndex(std::get<1>(indices)) {}

void VulkanTimerQuery::setFence(std::shared_ptr<VulkanCmdFence> fence) noexcept {
    std::unique_lock<utils::Mutex> lock(mFenceMutex);
    mFence = fence;
}

bool VulkanTimerQuery::isCompleted() noexcept {
    std::unique_lock<utils::Mutex> lock(mFenceMutex);
    // QueryValue is a synchronous call and might occur before beginTimerQuery has written anything
    // into the command buffer, which is an error according to the validation layer that ships in
    // the Android NDK.  Even when AVAILABILITY_BIT is set, validation seems to require that the
    // timestamp has at least been written into a processed command buffer.

    // This fence indicates that the corresponding buffer has been completed.
    if (!mFence) {
        return false;
    }
    VkResult status = mFence->status.load(std::memory_order_relaxed);
    if (status != VK_SUCCESS) {
        return false;
    }

    return true;
}

VulkanTimerQuery::~VulkanTimerQuery() = default;

VulkanRenderPrimitive::VulkanRenderPrimitive(VulkanResourceAllocator* resourceAllocator,
        PrimitiveType pt, Handle<HwVertexBuffer> vbh, Handle<HwIndexBuffer> ibh)
        : VulkanResource(VulkanResourceType::RENDER_PRIMITIVE),
          mResources(resourceAllocator) {
    type = pt;
    vertexBuffer = resourceAllocator->handle_cast<VulkanVertexBuffer*>(vbh);
    indexBuffer = resourceAllocator->handle_cast<VulkanIndexBuffer*>(ibh);
    mResources.acquire(vertexBuffer);
    mResources.acquire(indexBuffer);
}

using Bitmask = VulkanDescriptorSetLayout::Bitmask;

inline Bitmask Bitmask::fromBackendLayout(descset::DescriptorSetLayout const& layout) {
    Bitmask mask;
    for (auto const& binding: layout.bindings) {
        switch (binding.type) {
            case descset::DescriptorType::UNIFORM_BUFFER: {
                if (binding.flags == descset::DescriptorFlags::DYNAMIC_OFFSET) {
                    mask.dynamicUbo |= fromStageFlags<UniformBufferBitmask>(binding.stageFlags,
                            binding.binding);
                } else {
                    mask.ubo |= fromStageFlags<UniformBufferBitmask>(binding.stageFlags,
                            binding.binding);
                }
                break;
            }
            case descset::DescriptorType::SAMPLER: {
                mask.sampler |= fromStageFlags<SamplerBitmask>(binding.stageFlags, binding.binding);
                break;
            }
            case descset::DescriptorType::INPUT_ATTACHMENT: {
                mask.inputAttachment |=
                        fromStageFlags<InputAttachmentBitmask>(binding.stageFlags, binding.binding);
                break;
            }
        }
    }
    return mask;
}


} // namespace filament::backend
