#!/usr/bin/env python3
import os
import subprocess
import sys
import socket
import re

def get_local_ip():
    """获取本地局域网IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu():
    clear_screen()
    print('=======================================')
    print('           TTS服务启动器')
    print('=======================================')
    print('1. 启动CPU版本')
    print('2. 启动GPU版本')
    print('3. 退出')
    print('=======================================')
    print()

def get_cuda_version():
    """获取CUDA版本"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            output = result.stdout
            # 从nvidia-smi输出中解析CUDA版本
            import re
            match = re.search(r'CUDA Version:\s*(\d+\.\d+)', output)
            if match:
                return match.group(1)
        return None
    except Exception:
        return None

def check_torch_cuda_available():
    """检查PyTorch是否支持CUDA"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_gpu_pytorch(cuda_version=None):
    """安装GPU版本的PyTorch"""
    print('\n=======================================')
    print('正在安装GPU版本PyTorch...')
    print('=======================================')
    
    if not cuda_version:
        cuda_version = get_cuda_version()
    
    if not cuda_version:
        print('无法检测CUDA版本，尝试使用稳定版本')
        cuda_major = '12'
    else:
        cuda_major = cuda_version.split('.')[0] if '.' in cuda_version else cuda_version
        print(f'检测到CUDA版本: {cuda_version}')
    
    # 尝试的CUDA版本列表（按优先级）
    cuda_versions_to_try = []
    
    # 如果检测到CUDA 13.x，先尝试CUDA 12.x（更稳定）
    if cuda_major == '13':
        print('注意：CUDA 13.x可能还没有对应的PyTorch版本，优先尝试CUDA 12.x')
        cuda_versions_to_try = ['cu121', 'cu118', 'cu124']
    elif cuda_major == '12':
        cuda_versions_to_try = ['cu121', 'cu124', 'cu118']
    else:
        cuda_versions_to_try = ['cu118', 'cu121', 'cu124']
    
    print('\n1. 卸载当前PyTorch...')
    uninstall_cmd = [sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y']
    subprocess.run(uninstall_cmd, capture_output=True)
    
    # 尝试多个CUDA版本
    for index_suffix in cuda_versions_to_try:
        index_url = f'https://download.pytorch.org/whl/{index_suffix}'
        print(f'\n2. 尝试安装PyTorch (CUDA: {index_suffix})...')
        print(f'使用PyTorch安装源: {index_url}')
        
        install_cmd = [
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', index_url
        ]
        
        print(f'执行命令: {" ".join(install_cmd)}')
        
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f'\n✅ GPU版本PyTorch安装成功！(CUDA: {index_suffix})')
            return True
        else:
            print(f'\n⚠️  CUDA {index_suffix} 版本安装失败，尝试下一个版本...')
    
    # 如果所有版本都失败了，尝试从PyPI安装（可能是CPU版本）
    print(f'\n所有CUDA版本都安装失败，尝试从PyPI安装...')
    install_cmd = [
        sys.executable, '-m', 'pip', 'install', 
        'torch', 'torchvision', 'torchaudio'
    ]
    
    result = subprocess.run(install_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f'\n⚠️  安装了CPU版本的PyTorch（GPU版本不可用）')
        return False
    else:
        print(f'\n❌ PyTorch安装失败！')
        print('错误信息:')
        print(result.stderr)
        return False

def install_dependencies(version_type):
    """安装依赖"""
    print(f'\n=======================================')
    print(f'正在安装{version_type}版本依赖...')
    print(f'=======================================')
    
    project_dir = os.path.dirname(__file__)
    
    if version_type == 'CPU':
        target_dir = os.path.join(project_dir, 'tts-project-cpu')
    else:
        target_dir = os.path.join(project_dir, 'tts-project-gpu')
    
    os.chdir(target_dir)
    print(f'当前目录: {os.getcwd()}')
    print(f'使用Python解释器: {sys.executable}')
    
    if version_type == 'GPU':
        print('\n检查PyTorch CUDA支持...')
        if not check_torch_cuda_available():
            print('当前PyTorch不支持CUDA，需要安装GPU版本PyTorch')
            if not install_gpu_pytorch():
                print('\n⚠️  GPU版本PyTorch安装失败，将使用CPU版本')
                input('按 Enter 键继续...')
        else:
            print('✅ 当前PyTorch已支持CUDA！')
    
    print('\n安装其他依赖...')
    print('使用国内镜像源安装依赖...')
    
    install_cmd = [
        sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', 
        '--user', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'
    ]
    
    print(f'执行命令: {" ".join(install_cmd)}')
    
    result = subprocess.run(install_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print('\n✅ 依赖安装成功！')
        if len(result.stdout) > 500:
            print(f'安装输出: {result.stdout[:500]}...')
        else:
            print(f'安装输出: {result.stdout}')
        return True
    else:
        print('\n❌ 依赖安装失败！')
        print('错误信息:')
        print(result.stderr)
        return False

def start_service(version_type):
    """启动服务"""
    local_ip = get_local_ip()
    
    if version_type == 'CPU':
        port = '8000'
    else:
        port = '8001'
    
    print(f'\n=======================================')
    print(f'启动{version_type}版本服务...')
    print(f'=======================================')
    print(f'服务地址: http://localhost:{port}')
    print(f'服务地址: http://{local_ip}:{port} (局域网访问)')
    print('按 Ctrl+C 停止服务')
    print()
    print('=======================================')
    print('服务启动中，请稍候...')
    print('=======================================')
    print()
    
    try:
        print('启动uvicorn服务...')
        serve_cmd = [
            sys.executable, '-m', 'uvicorn', 'app.main:app', 
            '--host', '0.0.0.0', '--port', port,
            '--reload'
        ]
        
        print(f'执行命令: {" ".join(serve_cmd)}')
        print()
        
        print('启动服务...')
        print('注意：服务将在当前窗口运行，按 Ctrl+C 停止服务')
        print('服务启动后，请在浏览器中访问上述地址')
        print()
        
        try:
            print('正在启动服务...')
            print('如果服务启动成功，您将看到 uvicorn 相关的日志输出')
            print('如果服务启动失败，将显示错误信息')
            print()
            
            process = subprocess.Popen(
                serve_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            print('服务输出:')
            print('-' * 60)
            
            try:
                for line in iter(process.stdout.readline, ''):
                    print(line.strip())
                    if 'Uvicorn running on' in line:
                        print(f'\n✅ 服务启动成功！')
                        print('服务已在以下地址运行:')
                        print(f'  - http://localhost:{port}')
                        print(f'  - http://{local_ip}:{port} (局域网)')
                        print('\n按 Ctrl+C 停止服务...')
            except KeyboardInterrupt:
                print('\n\n正在停止服务...')
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print('服务已停止')
                input('按 Enter 键返回主菜单...')
            
            process.wait()
            if process.returncode != 0:
                print(f'\n服务异常退出，返回码: {process.returncode}')
                print('服务启动失败！')
                input('按 Enter 键返回主菜单...')
        except KeyboardInterrupt:
            print('\n服务已手动停止')
            input('按 Enter 键返回主菜单...')
    except KeyboardInterrupt:
        print('\n服务已停止')
    except Exception as e:
        print(f'服务启动出错: {e}')
        import traceback
        traceback.print_exc()
        input('按 Enter 键返回主菜单...')

def start_cpu():
    """启动CPU版本"""
    print('=======================================')
    print('启动CPU版本服务...')
    print('=======================================')
    
    if not install_dependencies('CPU'):
        input('按 Enter 键返回主菜单...')
        return
    
    start_service('CPU')

def start_gpu():
    """启动GPU版本"""
    print('=======================================')
    print('启动GPU版本服务...')
    print('=======================================')
    
    if not install_dependencies('GPU'):
        input('按 Enter 键返回主菜单...')
        return
    
    start_service('GPU')

def main():
    while True:
        show_menu()
        choice = input('请输入选项编号: ')
        
        if choice == '1':
            start_cpu()
        elif choice == '2':
            start_gpu()
        elif choice == '3':
            print('退出...')
            break
        else:
            print('错误: 无效的选项')
            input('按 Enter 键返回...')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test-gpu':
        print('进入测试模式，直接启动GPU版本服务...')
        start_gpu()
    elif len(sys.argv) > 1 and sys.argv[1] == '--test-cpu':
        print('进入测试模式，直接启动CPU版本服务...')
        start_cpu()
    else:
        main()