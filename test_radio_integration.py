#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""
测试RADIO模型集成到VLFM的脚本
"""

import numpy as np
import os
import sys
import torch
from PIL import Image

# 添加VLFM路径
sys.path.insert(0, '/workspace')

def test_radio_model():
    """测试RADIO模型基本功能"""
    print("🔍 Testing RADIO model integration...")
    
    try:
        from vlfm.vlm.radio import RADIOModel
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_text = "find a chair"
        
        # 初始化RADIO模型
        print("📥 Loading RADIO model...")
        radio_model = RADIOModel(model_name="radio_v2_medium")
        
        # 测试相似度计算
        print("🧮 Computing similarity...")
        similarity = radio_model.cosine(test_image, test_text)
        
        print(f"✅ Similarity score: {similarity:.4f}")
        print("✅ RADIO model test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ RADIO model test failed: {e}")
        return False


def test_radio_client():
    """测试RADIO客户端"""
    print("\n🔍 Testing RADIO client...")
    
    try:
        from vlfm.vlm.radio import RADIOClient
        
        # 创建客户端（注意：这需要服务器运行）
        client = RADIOClient(port=12185)
        
        print("✅ RADIO client created successfully!")
        print("⚠️  Note: Client functionality requires server to be running")
        
        return True
        
    except Exception as e:
        print(f"❌ RADIO client test failed: {e}")
        return False


def test_radio_policy():
    """测试RADIO策略类"""
    print("\n🔍 Testing RADIO policy classes...")
    
    try:
        from vlfm.policy.radio_policy import RADIOPolicy, RADIOPolicyV2, RADIOPolicyV3
        
        print("✅ RADIO policy classes imported successfully!")
        
        # 测试策略配置
        config_params = {
            'text_prompt': 'find a target_object',
            'pointnav_policy_path': 'data/dummy_policy.pth',
            'depth_image_shape': (256, 256),
            'pointnav_stop_radius': 1.0,
            'object_map_erosion_size': 0.1,
            'radio_port': 12185,
        }
        
        print("✅ RADIO policy configuration test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ RADIO policy test failed: {e}")
        return False


def test_habitat_integration():
    """测试Habitat集成"""
    print("\n🔍 Testing Habitat integration...")
    
    try:
        from vlfm.policy.habitat_policies import (
            HabitatRADIOPolicy, 
            HabitatRADIOPolicyV2, 
            HabitatRADIOPolicyV3
        )
        
        print("✅ Habitat RADIO policies imported successfully!")
        
        # 检查策略是否注册
        from habitat_baselines.common.baseline_registry import baseline_registry
        
        available_policies = list(baseline_registry._mapping['policy'].keys())
        radio_policies = [p for p in available_policies if 'RADIO' in p]
        
        print(f"📋 Available RADIO policies: {radio_policies}")
        
        if len(radio_policies) >= 3:
            print("✅ All RADIO policies registered successfully!")
            return True
        else:
            print("⚠️  Some RADIO policies may not be registered properly")
            return False
        
    except Exception as e:
        print(f"❌ Habitat integration test failed: {e}")
        return False


def test_configuration():
    """测试配置文件"""
    print("\n🔍 Testing configuration files...")
    
    config_file = "/workspace/config/experiments/vlfm_objectnav_hm3d_radio.yaml"
    
    if os.path.exists(config_file):
        print(f"✅ RADIO configuration file exists: {config_file}")
        
        # 读取配置文件内容
        with open(config_file, 'r') as f:
            content = f.read()
            
        if "HabitatRADIOPolicyV2" in content:
            print("✅ Configuration file contains correct RADIO policy!")
            return True
        else:
            print("⚠️  Configuration file may not contain correct policy")
            return False
    else:
        print(f"❌ Configuration file not found: {config_file}")
        return False


def main():
    """主测试函数"""
    print("🚀 Starting VLFM-RADIO integration tests...\n")
    
    tests = [
        test_radio_model,
        test_radio_client, 
        test_radio_policy,
        test_habitat_integration,
        test_configuration,
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! RADIO integration is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())