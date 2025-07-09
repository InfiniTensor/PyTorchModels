import os
import sys
import importlib.util
from importlib.abc import MetaPathFinder

# --- 为了避免魔法行为，提供清晰的日志 ---
print(f"--- [usercustomize.py v2.0 'Import Hook' in '{__file__}'] ---")
print(">>> 智能适配钩子已准备就绪，等待 'torch' 导入...")

class PlatformPatcher(MetaPathFinder):
    """
    一个元路径查找器（导入钩子），用于在`torch`被导入后自动加载国产适配库
    """
    # 此类变量用于确保补丁逻辑只执行一次，防止无限循环
    _patch_applied = False

    def find_spec(self, fullname, path, target=None):
        """
        Python 的 import 语句会调用此方法
        fullname: 正在被导入的模块的全名，例如 'torch' 或 'torch.nn'
        """
        # 1. 检测有没有 torch 导入
        # 我们只关心根模块 'torch' 的导入，并且只处理一次
        if not self.__class__._patch_applied and fullname == 'torch':

            # 找到 'torch', 现在我们接管它的加载过程。
            # 首先，我们需要找到原始的 'torch' 模块信息 (spec)
            # 为此，我们暂时将自己的钩子移除，以防无限递归调用 find_spec
            finder = sys.meta_path.pop(0)
            spec = importlib.util.find_spec(fullname)
            # 找到后，立刻把钩子加回去
            sys.meta_path.insert(0, finder)

            if spec:
                # 标记为已处理
                self.__class__._patch_applied = True
                print(f"\n>>> HOOK: 截获到 'import torch' 请求！正在处理...")

                # 关键：我们不直接返回原始 spec，而是包装它的加载器
                # 这样我们就能在模块执行后注入自己的逻辑
                original_loader_exec = spec.loader.exec_module
                spec.loader.exec_module = lambda module: self.execute_with_patch(original_loader_exec, module)
                return spec

        # 如果不是 'torch' 或者已经处理过了，我们什么都不做，交还给 Python 的标准导入器
        return None

    def execute_with_patch(self, original_loader_exec, module):
        """
        先执行原始的模块加载，然后根据检测到的国产硬件平台应用相应的补丁
        """
        # 1. 执行原始的 `torch` 模块加载
        original_loader_exec(module)
        print(f">>> HOOK: 'torch' v{module.__version__} 已成功加载")

        # 2. 检测并适配不同的国产硬件平台
        platform_env = os.environ.get('PLATFORM_ENV')

        # 尝试获取 CUDA_VISIBLE_DEVICES 的值
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if platform_env == 'ASCEND_NPU':

            print(">>> HOOK: 检测到昇腾（ASCEND_NPU）平台...")
            # 3.a 导入昇腾相关的库
            try:
                import torch_npu
                from torch_npu.contrib import transfer_to_npu
                
                # 检查获取到的值是否存在且不为空字符串
                if cuda_devices:
                    # 如果 cuda_devices 不是 None 且不是空字符串，就执行赋值操作
                    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = cuda_devices
                    print(f"成功将 ASCEND_RT_VISIBLE_DEVICES 设置为: {cuda_devices}")
                else:
                    print("环境变量 CUDA_VISIBLE_DEVICES 未设置或为空，跳过赋值。")

                print(">>> HOOK: 成功导入 'torch_npu'。NPU环境适配完成！")

            except ImportError:
                print(">>> HOOK: 警告！平台适配失败，无法导入 'torch_npu'")
            except Exception as e:
                print(f">>> HOOK: 错误！导入 'torch_npu' 时发生异常: {e}")

        elif platform_env == 'CAMBRICON_MLU':

            print(">>> HOOK: 检测到寒武纪（CAMBRICON_MLU）平台...")
            # 3.b 导入寒武纪相关的库
            try:
                import torch_mlu
                from torch_mlu.utils.model_transfer import transfer
                
                # 检查获取到的值是否存在且不为空字符串
                if cuda_devices:
                    # 如果 cuda_devices 不是 None 且不是空字符串，就执行赋值操作
                    os.environ["MLU_VISIBLE_DEVICES"] = cuda_devices
                    print(f"成功将 MLU_VISIBLE_DEVICES 设置为: {cuda_devices}")
                else:
                    print("环境变量 CUDA_VISIBLE_DEVICES 未设置或为空，跳过赋值。")
                
                print(">>> HOOK: 成功导入 'torch_mlu'。MLU环境适配完成！")
            except ImportError:
                print(">>> HOOK: 警告！平台适配失败，无法导入 'torch_mlu'")
            except Exception as e:
                print(f">>> HOOK: 错误！导入 'torch_mlu' 时发生异常: {e}")

        else:
            print(">>> HOOK: 非昇腾或寒武纪平台，跳过特定硬件库的导入。")

        # 将加载好的 torch 模块放入 sys.modules，这是 import 机制的一部分
        sys.modules['torch'] = module

def install_hook():
    """
    将我们的自定义钩子安装到 Python 导入系统的最前面，确保最高优先级。
    """
    # 检查是否已安装，避免重复安装
    if not any(isinstance(p, PlatformPatcher) for p in sys.meta_path):
        sys.meta_path.insert(0, PlatformPatcher())

# 在 usercustomize.py 被 Python 加载时，立即安装我们的钩子
install_hook()

