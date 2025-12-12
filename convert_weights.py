"""
权重转换脚本：将旧的FingerprintGate模型权重转换为FingerprintFusion格式

使用方法：
    python convert_weights.py --input_path ./weight/old_model.pt --output_path ./weight/new_model.pt

功能：
- 从旧模型中提取fingerprint_gate.fp_proj权重
- 复制到新的fingerprint_fusion.fp_proj
- 初始化fingerprint_fusion.fusion_proj为Xavier初始化
- 保持其他权重不变
"""

import torch
import argparse
import os


def convert_fingerprint_gate_to_fusion(state_dict, verbose=True):
    """
    将旧的FingerprintGate权重转换为FingerprintFusion格式

    Args:
        state_dict: 旧模型的state_dict
        verbose: 是否打印详细信息

    Returns:
        转换后的state_dict
    """
    new_state_dict = {}
    converted_keys = []

    for key, value in state_dict.items():
        new_key = key

        # 转换fingerprint_gate -> fingerprint_fusion
        if 'fingerprint_gate.fp_proj' in key:
            new_key = key.replace('fingerprint_gate.fp_proj', 'fingerprint_fusion.fp_proj')
            converted_keys.append(f"{key} -> {new_key}")
        elif 'fingerprint_gate.gate' in key:
            # 跳过gate权重（不再需要）
            if verbose:
                print(f"  ⚠️  跳过: {key} (Gate机制已移除)")
            continue
        elif 'fingerprint_gate.q_proj' in key:
            # 跳过注意力权重（不再需要）
            if verbose:
                print(f"  ⚠️  跳过: {key} (注意力机制已移除)")
            continue
        elif 'fingerprint_gate.attn_blend' in key:
            # 跳过注意力参数
            if verbose:
                print(f"  ⚠️  跳过: {key} (注意力参数已移除)")
            continue

        new_state_dict[new_key] = value

    # 初始化fusion_proj权重（Xavier初始化）
    # 注意：这里假设hidden_dim=128（根据配置文件）
    # 如果不同，需要从实际模型中获取
    for key in new_state_dict.keys():
        if 'fingerprint_fusion.fp_proj.3.weight' in key:
            # 找到了fp_proj的最后一层，说明hidden_dim可以从这里推断
            hidden_dim = new_state_dict[key].shape[0]

            # 初始化fusion_proj.weight: [hidden_dim, 2*hidden_dim]
            fusion_weight = torch.empty(hidden_dim, 2 * hidden_dim)
            torch.nn.init.xavier_uniform_(fusion_weight)
            new_state_dict['fingerprint_fusion.fusion_proj.weight'] = fusion_weight

            # 初始化fusion_proj.bias: [hidden_dim]
            fusion_bias = torch.zeros(hidden_dim)
            new_state_dict['fingerprint_fusion.fusion_proj.bias'] = fusion_bias

            if verbose:
                print(f"\n✅ 新增权重:")
                print(f"  - fingerprint_fusion.fusion_proj.weight: {fusion_weight.shape}")
                print(f"  - fingerprint_fusion.fusion_proj.bias: {fusion_bias.shape}")
            break

    if verbose and converted_keys:
        print(f"\n✅ 转换的权重 ({len(converted_keys)}个):")
        for conv in converted_keys:
            print(f"  - {conv}")

    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description='转换FingerprintGate权重为FingerprintFusion格式')
    parser.add_argument('--input_path', type=str, required=True, help='旧模型权重路径')
    parser.add_argument('--output_path', type=str, required=True, help='新模型权重输出路径')
    parser.add_argument('--quiet', action='store_true', help='静默模式（不打印详细信息）')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input_path):
        print(f"❌ 错误：输入文件不存在: {args.input_path}")
        return

    # 加载旧模型
    print(f"📂 加载旧模型: {args.input_path}")
    state_dict = torch.load(args.input_path, map_location='cpu')

    # 统计权重信息
    total_keys = len(state_dict)
    fp_gate_keys = sum(1 for k in state_dict.keys() if 'fingerprint_gate' in k)

    print(f"  - 总权重数: {total_keys}")
    print(f"  - fingerprint_gate相关: {fp_gate_keys}")

    # 执行转换
    print(f"\n🔄 开始转换...")
    new_state_dict = convert_fingerprint_gate_to_fusion(state_dict, verbose=not args.quiet)

    # 统计新权重信息
    new_total_keys = len(new_state_dict)
    fp_fusion_keys = sum(1 for k in new_state_dict.keys() if 'fingerprint_fusion' in k)

    print(f"\n📊 转换统计:")
    print(f"  - 原权重数: {total_keys}")
    print(f"  - 新权重数: {new_total_keys}")
    print(f"  - fingerprint_fusion相关: {fp_fusion_keys}")

    # 保存新模型
    print(f"\n💾 保存新模型: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    torch.save(new_state_dict, args.output_path)

    # 验证
    print(f"\n✅ 验证保存...")
    loaded = torch.load(args.output_path, map_location='cpu')
    assert len(loaded) == new_total_keys, "权重数量不匹配！"

    print(f"\n🎉 转换完成！")
    print(f"   输入: {args.input_path}")
    print(f"   输出: {args.output_path}")
    print(f"\n使用方法:")
    print(f"   修改配置文件中的model_path为: {args.output_path}")
    print(f"   然后运行: python generate.py")


if __name__ == '__main__':
    main()
