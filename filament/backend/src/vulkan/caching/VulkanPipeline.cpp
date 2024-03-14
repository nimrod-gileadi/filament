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

#include "VulkanPipeline.h"

namespace filament::backend {

VkPipelineLayout VulkanPipelineLayoutCache::getLayout(
        VulkanDescriptorSetManager::LayoutArray const& descriptorSetLayouts) {
    // VkPipelineLayout VulkanPipelineLayoutCache::getLayout(PipelineLayoutKey const& key) {

    // TODO: maybe we can do better here then copying here.
    PipelineLayoutKey key;
    for (uint8_t i = 0; i < descriptorSetLayouts.size(); ++i) {
        key[i] = descriptorSetLayouts[i];
    }

    auto iter = mPipelineLayouts.find(key);
    if (iter != mPipelineLayouts.end()) {
        PipelineLayoutCacheEntry& entry = mPipelineLayouts[key];
        entry.lastUsed = mTimestamp++;
        return entry.handle;
    }

    VkPipelineLayoutCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = descriptorSetLayouts.size(),
        .pSetLayouts = key.data(),
        .pushConstantRangeCount = 0,
    };
    VkPipelineLayout layout;
    vkCreatePipelineLayout(mDevice, &info, VKALLOC, &layout);

    mPipelineLayouts[key] = {
        .handle = layout,
        .lastUsed = mTimestamp++,
    };
    return layout;
}

size_t VulkanPipelineLayoutCache::PipelineLayoutKeyHashFn::operator()(
        const PipelineLayoutKey& key) const {
    std::hash<VkDescriptorSetLayout> hasher;
    auto h0 = hasher(key[0]);
    auto h1 = hasher(key[1]);
    auto h2 = hasher(key[2]);

    // TODO: Probably not the right way to combine hashes.
    return (h0 ^ (h1 << 1)) ^ (h2 << 1);
}

bool VulkanPipelineLayoutCache::PipelineLayoutKeyEqual::operator()(const PipelineLayoutKey& k1,
        const PipelineLayoutKey& k2) const {
    return k1 == k2;
}

}// namespace filament::backend
