import os
import sys
import types
import importlib.util
from importlib.abc import MetaPathFinder

# --- 为了避免魔法行为，提供清晰的日志 ---
print(f"--- [usercustomize.py v3.0 'Unified 7-Platform Import Hook' in '{__file__}'] ---")
print(">>> 统一适配钩子已准备就绪，等待 'torch' 导入...")

# --- 全局：cambricon (torchnet) 依赖 visdom，但部分平台环境装不上 visdom ---
# 已安装 visdom 的环境不受影响（先尝试 import）；未安装时提供一个空 mock，避免 ImportError
if 'visdom' not in sys.modules:
    try:
        import visdom  # noqa: F401
    except ImportError:
        sys.modules['visdom'] = types.ModuleType('visdom')

# 当前目标平台（合法取值见 env.sh 注释）：
#   NVIDIA_GPU / ASCEND_NPU / CAMBRICON_MLU / HYGON_DCU(=SUGON_DCU) /
#   ILLUVATAR_GPU / METAX_GPU / MOORE_GPU
_PLATFORM = os.environ.get('PLATFORM_ENV', 'NVIDIA_GPU')


class PlatformPatcher(MetaPathFinder):
    """
    一个元路径查找器（导入钩子）：在 `torch` 被导入后，按 PLATFORM_ENV
    自动加载对应的国产硬件适配库，并（仅 MOORE 时）把 torch.cuda 全套
    重定向到 torch_musa。

    平台适配策略：
      - ASCEND_NPU   : import torch_npu + transfer_to_npu
      - CAMBRICON_MLU: import torch_mlu + transfer
      - MOORE_GPU    : import torch_musa + overwrite_cuda_api + cuda->musa 全套重定向
      - 其余(NVIDIA/HYGON_DCU/SUGON_DCU/ILLUVATAR_GPU/METAX_GPU): 纯 CUDA 兼容透传
        （这些厂商库自身提供 CUDA 兼容层，torch.cuda 直接可用，无需导入额外库）
    """
    _patch_applied = False

    def find_spec(self, fullname, path, target=None):
        # 只关心根模块 'torch' 的导入，且只处理一次
        if not self.__class__._patch_applied and fullname == 'torch':
            # 找到 'torch'，接管它的加载过程。
            # 临时把自己的钩子移除，以防无限递归调用 find_spec
            finder = sys.meta_path.pop(0)
            spec = importlib.util.find_spec(fullname)
            sys.meta_path.insert(0, finder)

            if spec:
                self.__class__._patch_applied = True
                print(f"\n>>> HOOK: 截获到 'import torch' 请求！当前平台: {_PLATFORM}")

                # 包装原始加载器，在模块执行后注入平台适配逻辑
                original_loader_exec = spec.loader.exec_module
                spec.loader.exec_module = lambda module: self.execute_with_patch(original_loader_exec, module)
                return spec

        return None

    def execute_with_patch(self, original_loader_exec, module):
        """先执行原始的 torch 加载，再按平台应用适配补丁"""
        original_loader_exec(module)
        print(f">>> HOOK: 'torch' v{module.__version__} loaded")

        # 统一以 CUDA_VISIBLE_DEVICES 作为可见设备的单一真源；
        # 各平台的原生设备变量（ASCEND_RT_/MLU_/MUSA_VISIBLE_DEVICES）
        # 由 env.sh 或下面的分支负责映射。
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        if _PLATFORM == 'MOORE_GPU':
            self._patch_moore(module)
        elif _PLATFORM == 'ASCEND_NPU':
            self._patch_ascend(cuda_devices)
        elif _PLATFORM == 'CAMBRICON_MLU':
            self._patch_cambricon(cuda_devices)
        else:
            # NVIDIA_GPU / HYGON_DCU / SUGON_DCU / ILLUVATAR_GPU / METAX_GPU
            # 纯 CUDA 兼容，不导入额外硬件库
            # 注：METAX 默认按纯 CUDA 处理（与 metax_change 一致）。若沐曦实测需
            #     走 musa 兼容层，把 'METAX_GPU' 加入上面的 _patch_moore 条件即可。
            print(f">>> HOOK: 平台 {_PLATFORM} 走纯 CUDA 兼容路径，跳过特定硬件库的导入。")

        sys.modules['torch'] = module

    # ---------------- ASCEND_NPU ----------------
    def _patch_ascend(self, cuda_devices):
        print(">>> HOOK: 检测到昇腾（ASCEND_NPU）平台...")
        try:
            import torch_npu
            from torch_npu.contrib import transfer_to_npu

            if cuda_devices:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = cuda_devices
                print(f"成功将 ASCEND_RT_VISIBLE_DEVICES 设置为: {cuda_devices}")
            else:
                print("CUDA_VISIBLE_DEVICES 未设置或为空，跳过设备变量赋值。")

            print(">>> HOOK: 成功导入 'torch_npu'。NPU 环境适配完成！")
        except ImportError:
            print(">>> HOOK: 警告！平台适配失败，无法导入 'torch_npu'（是否已 source 昇腾 set_env.sh？）")
        except Exception as e:
            print(f">>> HOOK: 错误！导入 'torch_npu' 时发生异常: {e}")

    # ---------------- CAMBRICON_MLU ----------------
    def _patch_cambricon(self, cuda_devices):
        print(">>> HOOK: 检测到寒武纪（CAMBRICON_MLU）平台...")
        try:
            import torch_mlu
            from torch_mlu.utils.model_transfer import transfer

            if cuda_devices:
                os.environ["MLU_VISIBLE_DEVICES"] = cuda_devices
                print(f"成功将 MLU_VISIBLE_DEVICES 设置为: {cuda_devices}")
            else:
                print("CUDA_VISIBLE_DEVICES 未设置或为空，跳过设备变量赋值。")

            print(">>> HOOK: 成功导入 'torch_mlu'。MLU 环境适配完成！")
        except ImportError:
            print(">>> HOOK: 警告！平台适配失败，无法导入 'torch_mlu'（是否已放置 sitecustomize.py 设 ENABLE_USER_SITE=True？）")
        except Exception as e:
            print(f">>> HOOK: 错误！导入 'torch_mlu' 时发生异常: {e}")

    # ---------------- MOORE_GPU (musa 全套重定向) ----------------
    def _patch_moore(self, module):
        print(">>> HOOK: 检测到摩尔线程（MOORE_GPU）平台...")

        # 仅 MOORE 才安装 transformers 的 torch.load 安全校验绕过（要求 torch>=2.6）
        self._install_transformers_patch()

        try:
            import torch_musa
            torch_musa.overwrite_cuda_api()

            # 代理 torch.cuda 的关键函数到 torch_musa
            module.cuda.is_available = torch_musa.is_available
            module.cuda.device_count = torch_musa.device_count
            module.cuda.current_device = torch_musa.current_device
            module.cuda.set_device = torch_musa.set_device
            module.cuda.synchronize = torch_musa.synchronize
            module.cuda.manual_seed = torch_musa.manual_seed
            module.cuda.manual_seed_all = torch_musa.manual_seed_all
            module.cuda.get_device_properties = torch_musa.get_device_properties
            module.cuda.get_device_name = torch_musa.get_device_name
            module.cuda.get_device_capability = torch_musa.get_device_capability
            module.cuda.memory_allocated = torch_musa.memory_allocated
            module.cuda.memory_reserved = torch_musa.memory_reserved
            module.cuda.max_memory_allocated = torch_musa.max_memory_allocated
            module.cuda.max_memory_reserved = torch_musa.max_memory_reserved
            module.cuda.empty_cache = torch_musa.empty_cache

            # 让 torch.device("cuda") 和 torch.device("cuda:0") 重定向到 "musa"
            _orig_device = module.device

            class _PatchedDevice:
                """拦截 torch.device("cuda[:X]") 调用，重定向到 musa"""
                def __init__(self, orig_device_cls):
                    self._orig = orig_device_cls

                def __call__(self, type_arg, index=None):
                    if isinstance(type_arg, str):
                        if type_arg == "cuda":
                            return self._orig("musa", index)
                        if type_arg.startswith("cuda:"):
                            idx = type_arg.split(":")[1]
                            return self._orig(f"musa:{idx}")
                    if isinstance(type_arg, int) and index is not None:
                        return self._orig(type_arg, index)
                    return self._orig(type_arg) if index is None else self._orig(type_arg, index)

                def __instancecheck__(self, instance):
                    return isinstance(instance, self._orig)

            module.device = _PatchedDevice(_orig_device)

            # 让 Tensor.cuda() 重定向到 Tensor.musa()
            def _patched_cuda(self_tensor, *args, **kwargs):
                return self_tensor.musa(*args, **kwargs)

            module.Tensor.cuda = _patched_cuda

            # 拦截 .to("cuda") / .to("cuda:0") 调用，将字符串替换为 "musa"
            def _cuda_to_musa_arg(args, kwargs):
                """将 args/kwargs 中的 "cuda" / "cuda:X" 替换为 "musa" / "musa:X" """
                new_args = []
                for a in args:
                    if isinstance(a, str):
                        if a == "cuda":
                            a = "musa"
                        elif a.startswith("cuda:"):
                            a = "musa:" + a[5:]
                    elif isinstance(a, _orig_device):
                        s = str(a)
                        if s.startswith("cuda"):
                            a = _orig_device(s.replace("cuda", "musa", 1))
                    new_args.append(a)
                if "device" in kwargs:
                    d = kwargs["device"]
                    if isinstance(d, str):
                        if d == "cuda":
                            kwargs["device"] = "musa"
                        elif d.startswith("cuda:"):
                            kwargs["device"] = "musa:" + d[5:]
                    elif isinstance(d, _orig_device):
                        s = str(d)
                        if s.startswith("cuda"):
                            kwargs["device"] = _orig_device(s.replace("cuda", "musa", 1))
                return tuple(new_args), kwargs

            _orig_module_to = module.nn.Module.to

            def _patched_module_to(self, *args, **kwargs):
                args, kwargs = _cuda_to_musa_arg(args, kwargs)
                return _orig_module_to(self, *args, **kwargs)

            module.nn.Module.to = _patched_module_to

            _orig_tensor_to = module.Tensor.to

            def _patched_tensor_to(self, *args, **kwargs):
                args, kwargs = _cuda_to_musa_arg(args, kwargs)
                return _orig_tensor_to(self, *args, **kwargs)

            module.Tensor.to = _patched_tensor_to

            print(">>> HOOK: 成功导入 'torch_musa'。摩尔线程环境适配完成！")

        except ImportError:
            print(">>> HOOK: 警告！平台适配失败，无法导入 'torch_musa'")
        except Exception as e:
            print(f">>> HOOK: 错误！导入 'torch_musa' 时发生异常: {e}")

    @staticmethod
    def _install_transformers_patch():
        """
        绕过 transformers 的 torch.load 安全检查（要求 torch>=2.6）。
        注册 import hook：当 transformers.utils.import_utils 加载后，立刻 patch 掉检查函数。
        """
        class _TfPatcher(MetaPathFinder):
            _patched = False

            def find_spec(self, fullname, path, target=None):
                if not self.__class__._patched and fullname == "transformers.utils.import_utils":
                    self.__class__._patched = True
                    self_ref = self
                    sys.meta_path.remove(self_ref)
                    try:
                        spec = importlib.util.find_spec(fullname)
                    finally:
                        sys.meta_path.append(self_ref)
                    if spec and spec.loader and hasattr(spec.loader, 'exec_module'):
                        _orig = spec.loader.exec_module

                        def _wrapped(mod):
                            _orig(mod)
                            if hasattr(mod, 'check_torch_load_is_safe'):
                                mod.check_torch_load_is_safe = lambda: None

                        spec.loader.exec_module = _wrapped
                    return spec
                return None

        sys.meta_path.insert(0, _TfPatcher())


def install_hook():
    """将自定义钩子安装到 Python 导入系统的最前面，确保最高优先级。"""
    if not any(isinstance(p, PlatformPatcher) for p in sys.meta_path):
        sys.meta_path.insert(0, PlatformPatcher())


# 在 usercustomize.py 被 Python 加载时，立即安装钩子
install_hook()
