//
// Created by dipet on 13.04.2020.
//

#ifndef PHOTORADARSOURCES_ANIMATION_H
#define PHOTORADARSOURCES_ANIMATION_H

#include <vector>
#include <string>
#include <limits>

#include <unordered_map>

#include "gltf_helpers.h"
#include "Matrix.h"
#include "Quaternion.h"

namespace sutil {
class Animation {
private:
    struct AnimationChannel {
        std::pair<float, float> time_min_max;
        GltfBuffer time_points;
        GltfBuffer values;
        std::string interpolation;
        int node;
        std::string target;

        AnimationChannel(const tinygltf::Model& model,
                         const tinygltf::AnimationChannel& channel,
                         const tinygltf::AnimationSampler& sampler)
            :
              time_points(model, sampler.input),
              values(model, sampler.output),
              interpolation(sampler.interpolation),
              node(channel.target_node),
              target(channel.target_path)
        {
            if(interpolation != "LINEAR") {
                LOGC(ERROR) << "Animation support only LINEAR interpolation." << std::endl;
                exit(1);
            }
            checkTypes();
            time_min_max = time_points.getMinMax<float>();
        }

        void checkTypes()
        {
            bool correct = time_points.typeIsSame<float>();
            if(target == "translation" || target == "scale") {
                correct &= values.typeIsSame<float3>();
            } else if(target == "rotation") {
                correct &= values.typeIsSame<float4>();
            } else {
                LOGC(ERROR) << "Animation does not supports target. Node:  " << node
                            << " Path: " << target << std::endl;
                exit(1);
            }

            if(!correct) {
                LOGC(ERROR) << "Animation unexpected type. Node: " << node << " Path: " << target << std::endl;
                exit(1);
            }
        }
    };

public:
    Animation(const tinygltf::Animation& animation, const tinygltf::Model& model)
        :
            name_(animation.name)
    {
        for(const auto& channel : animation.channels) {
            const auto& sampler = animation.samplers[channel.sampler];
            AnimationChannel save_channel(model, channel, sampler);
            std::unordered_map<int, AnimationChannel>* channels = nullptr;

            if(save_channel.target == "translation") {
                channels = &translations_;
            } else if(save_channel.target == "scale") {
                channels = &scales_;
            } else if(save_channel.target == "rotation") {
                channels = &rotations_;
            } else {
                LOGC(ERROR) << "Unsupported target " << save_channel.target << std::endl;
                exit(1);
            }
            if(channels->find(save_channel.node) != channels->end()) {
                LOGC(ERROR) << "Few channels per node " << channel.target_node << std::endl;
                exit(1);
            }
            time_min_max_.first = std::min(time_min_max_.first, save_channel.time_min_max.first);
            time_min_max_.second = std::max(time_min_max_.first, save_channel.time_min_max.second);
            channels->emplace(make_pair(save_channel.node, std::move(save_channel)));
        }
    }

    inline std::pair<float, float> timeMinMax() const { return time_min_max_; }

    std::vector<int> getNodes() const;

    inline bool hasRotation(int node) const { return rotations_.find(node) != rotations_.end(); }
    inline bool hasTranslate(int node) const { return translations_.find(node) != translations_.end(); }
    inline bool hasScale(int node) const { return scales_.find(node) != scales_.end(); }

    inline Matrix4x4 rotation(int node, float time) const { return getMatrix(node, time, rotations_); }
    inline Matrix4x4 translate(int node, float time) const { return getMatrix(node, time, translations_); }
    inline Matrix4x4 scale(int node, float time) const { return getMatrix(node, time, scales_); }

private:
    static Matrix4x4 computeLinear(float t, float t1, float t2, const Matrix4x4& v1, const Matrix4x4& v2);
    static Matrix4x4 getMatrix(int node, float time, const std::unordered_map<int, AnimationChannel>& channels);

private:
    std::string name_;

    std::pair<float, float> time_min_max_ = { std::numeric_limits<float>::max(), std::numeric_limits<float>::min() };
    std::unordered_map<int, AnimationChannel> translations_;
    std::unordered_map<int, AnimationChannel> scales_;
    std::unordered_map<int, AnimationChannel> rotations_;
};
}

#endif //PHOTORADARSOURCES_ANIMATION_H
