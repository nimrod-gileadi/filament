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

#ifndef TNT_FILAMENT_BACKEND_VULKANPIPELINE_H
#define TNT_FILAMENT_BACKEND_VULKANPIPELINE_H

#include <vulkan/caching/VulkanDescriptorSet.h>

#include <bluevk/BlueVK.h>

namespace filament::backend {

class VulkanPipelineLayoutCache {
public:
    VulkanPipelineLayoutCache(VkDevice device)
        : mDevice(device),
          mTimestamp(0) {}

    using PipelineLayoutKey = std::array<VkDescriptorSetLayout,
            VulkanDescriptorSetManager::DISTINCT_DESCRIPTOR_SET_COUNT>;

    VulkanPipelineLayoutCache(VulkanPipelineLayoutCache const&) = delete;
    VulkanPipelineLayoutCache& operator=(VulkanPipelineLayoutCache const&) = delete;

    VkPipelineLayout getLayout(
            VulkanDescriptorSetManager::LayoutArray const& ldescriptorSetLayouts);
    //    VkPipelineLayout getLayout(PipelineLayoutKey const& key);

private:
    using Timestamp = uint64_t;
    struct PipelineLayoutCacheEntry {
        VkPipelineLayout handle;
        Timestamp lastUsed;
    };

    struct PipelineLayoutKeyHashFn {
        size_t operator()(PipelineLayoutKey const& key) const;
    };

    struct PipelineLayoutKeyEqual {
        bool operator()(PipelineLayoutKey const& k1, PipelineLayoutKey const& k2) const;
    };

    using PipelineLayoutMap = tsl::robin_map<PipelineLayoutKey , PipelineLayoutCacheEntry,
            PipelineLayoutKeyHashFn, PipelineLayoutKeyEqual>;
    
    VkDevice mDevice;
    Timestamp mTimestamp;
    PipelineLayoutMap mPipelineLayouts;
};

}

#endif // TNT_FILAMENT_BACKEND_VULKANPIPELINE_H
