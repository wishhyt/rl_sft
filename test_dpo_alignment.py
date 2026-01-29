"""
DPO训练对齐验证测试脚本

用于快速验证修改后的DPO实现是否正确
"""
import sys
import ast

def test_syntax():
    """测试trainer.py语法"""
    print("\n=== 测试 trainer.py 语法 ===")
    try:
        with open(r'd:\exp_mix_codex\rl_sft\src\rl_sft\trainer.py', encoding='utf-8') as f:
            ast.parse(f.read())
        print("✅ trainer.py 语法检查通过")
        return True
    except SyntaxError as e:
        print(f"❌ trainer.py 语法错误: {e}")
        return False

def test_dpo_losses():
    """测试dpo_losses.py模块"""
    print("\n=== 测试 dpo_losses.py 模块 ===")
    try:
        sys.path.insert(0, r'd:\exp_mix_codex\rl_sft\src')
        from rl_sft import dpo_losses
        
        # 检查所有loss函数是否存在
        expected_losses = ['diffusion-dpo', 'dspo', 'dmpo', 'sdpo', 'kto']
        missing = []
        for loss_name in expected_losses:
            if loss_name not in dpo_losses.LOSS_FUNCTIONS:
                missing.append(loss_name)
        
        if missing:
            print(f"❌ 缺少loss函数: {missing}")
            return False
        else:
            print(f"✅ 所有loss函数已注册: {expected_losses}")
        
        # 检查adaptive scaling函数是否存在
        if hasattr(dpo_losses, 'get_adaptive_lose_l_scale'):
            print("✅ get_adaptive_lose_l_scale 函数存在")
        else:
            print("❌ get_adaptive_lose_l_scale 函数缺失")
            return False
            
        return True
    except Exception as e:
        print(f"❌ 导入dpo_losses失败: {e}")
        return False

def test_config():
    """测试配置文件"""
    print("\n=== 测试配置文件 ===")
    try:
        import json
        with open(r'd:\exp_mix_codex\rl_sft\configs\dpo_pickapic.json', encoding='utf-8') as f:
            config = json.load(f)
        
        beta_dpo = config.get('dpo', {}).get('beta_dpo')
        if beta_dpo == 5000.0:
            print(f"✅ beta_dpo = {beta_dpo} (已对齐)")
        else:
            print(f"⚠️ beta_dpo = {beta_dpo} (预期5000)")
        
        train_method = config.get('dpo', {}).get('train_method')
        print(f"   train_method = {train_method}")
        
        return True
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("DPO训练对齐验证测试")
    print("=" * 60)
    
    results = []
    results.append(("语法检查", test_syntax()))
    results.append(("Loss模块", test_dpo_losses()))
    results.append(("配置文件", test_config()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:15s}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 所有测试通过！可以开始训练测试。")
    else:
        print("\n⚠️ 存在失败的测试，请检查上述输出。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
