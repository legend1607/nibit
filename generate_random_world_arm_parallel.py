import os
import json
import time
import random
import numpy as np
import pybullet as p
from os.path import join
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method, Manager
try:
    from path_planning_classes_arm.bit_star import BITStar
except Exception as e:
    print("捕获到异常:", e, flush=True)
# ----------------- NumPy 类型转换 -----------------
def np_to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, list):
        return [np_to_python(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: np_to_python(v) for k, v in obj.items()}
    else:
        return obj

# ----------------- 随机障碍物 -----------------
def add_random_obstacles(env, config):
    obstacles = []
    x_range, y_range, z_range = config["env_range"]
    num_obstacles = random.randint(*config["num_boxes_range"])
    
    for _ in range(num_obstacles):
        size_x = random.uniform(*config["box_size_range"])
        size_y = random.uniform(*config["box_size_range"])
        size_z = random.uniform(*config["box_size_range"])
        half_extents = [size_x/2, size_y/2, size_z/2]
        
        max_attempts = 10
        for _ in range(max_attempts):
            pos = [random.uniform(*x_range), 
                   random.uniform(*y_range), 
                   random.uniform(*z_range)]
            if abs(pos[0]) >= 0.5 and abs(pos[1]) >= 0.5:
                break
        
        env.add_box_obstacle(half_extents, pos)
        obstacles.append(("box", half_extents, pos))
    
    return obstacles

# ----------------- 直线路径判断（四维关节角度，更严格） -----------------
def is_straight_path(path, th=1.05):  # ✅ 从1.10改为1.05
    """
    判断机械臂关节轨迹是否接近直线。
    对于每个关节维度，计算实际路径长度与直线距离的比值。
    """
    path = np.array(path)  # shape: [T, D]
    T, D = path.shape

    if T < 2:
        return True

    for dim in range(D):
        joint_path = path[:, dim]

        # 计算相邻点的角度差（wrap到[-π, π]）
        diffs = joint_path[1:] - joint_path[:-1]
        diffs = (diffs + np.pi) % (2*np.pi) - np.pi

        # 实际轨迹长度（累积角度变化）
        L = np.sum(np.abs(diffs))

        # 起点到终点的最短角距离
        D_joint = joint_path[-1] - joint_path[0]
        D_joint = (D_joint + np.pi) % (2*np.pi) - np.pi
        D_joint = abs(D_joint)

        if D_joint < 1e-6:
            continue  # 该关节几乎不动

        # 判断路径曲折程度
        if (L / D_joint) > th:
            return False  # 任一关节偏离直线 → 非直线路径

    return True

# ----------------- 单个环境生成（带超时和错误处理） -----------------
def generate_single_env(args):
    task_id, config, shared_stats, TARGET, timeout = args
    N = config["num_samples_per_env"]
    max_env_retries = 5
    max_path_retries = 10

    cid = p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    def safe_disconnect():
        try:
            if p.isConnected(cid):
                p.disconnect(cid)
        except:
            pass

    start_env_time = time.time()
    
    for env_retry in range(max_env_retries):
        # ✅ 检查总超时
        if time.time() - start_env_time > timeout:
            safe_disconnect()
            return {"ok": False, "timeout": True}

        try:
            if config["env_type"] == "ur5":
                from environment.ur5_env import UR5Env
                env = UR5Env(GUI=False)
            elif config["env_type"] == "kuka":
                from environment.kuka_env import KukaEnv
                env = KukaEnv(GUI=False)
            else:
                from environment.liche_env import LicheEnv
                env = LicheEnv(GUI=False)
        except Exception as e:
            if env_retry == max_env_retries - 1:
                safe_disconnect()
                return {"ok": False, "error": f"Failed to create env: {e}"}
            continue

        obstacles = add_random_obstacles(env, config)

        # ✅ 环境可用性检查
        test_samples_needed = 5
        test_samples_success = 0
        for _ in range(test_samples_needed * 3):
            try:
                test_point = env.sample_empty_points()
                if test_point is not None:
                    test_samples_success += 1
                    if test_samples_success >= test_samples_needed:
                        break
            except:
                continue

        if test_samples_success < test_samples_needed:
            try:
                env.close()
            except:
                pass
            continue  # 环境太拥挤，重试

        # ✅ 生成路径
        paths, starts, goals = [], [], []
        path_gen_start = time.time()

        while len(paths) < N:
            # 检查路径生成超时
            if time.time() - path_gen_start > timeout * 0.8:  # 留20%余量
                break
            
            success = False
            for _ in range(max_path_retries):
                if time.time() - start_env_time > timeout:
                    break
                
                try:
                    problem = env.set_random_init_goal()
                    start, goal = problem["start"], problem["goal"]
                    if start is None or goal is None:
                        continue
                    planner = BITStar(
                        start=start,
                        goal=goal,
                        environment=env,
                        iter_max=1000,        # ✅ 平衡：400→500
                        batch_size=150,      # ✅ 平衡：100→150
                        pc_n_points=config.get("pc_n_points", 2048),
                        plot_flag=False
                    )

                    path_start_time = time.time()
                    try:
                        planner.planning(visualize=False)
                        path = planner.get_best_path()
                    except (TypeError, AttributeError, RuntimeError) as e:
                        error_msg = str(e)
                        if "NoneType" in error_msg or "sample_empty_points" in error_msg or "too crowded" in error_msg:
                            # 环境问题，放弃这个环境
                            break
                        raise
                    
                    elapsed = time.time() - path_start_time
                    if elapsed > 40:
                        continue  # 单条路径太慢，跳过

                    if path is None or len(path) == 0:
                        continue

                    paths.append(path)
                    starts.append(start)
                    goals.append(goal)
                    success = True
                    break
                    
                except Exception as e:
                    import traceback
                    print("捕获到异常:", e)
                    traceback.print_exc()
                    continue

            if not success:
                break  # 失败太多次，放弃当前环境

        # ✅ 只有成功生成N条路径才接受
        if len(paths) == N:
            n_straight = sum([is_straight_path(p) for p in paths])

            with shared_stats['lock']:
                total_paths = shared_stats['total_paths']
                straight_paths = shared_stats['straight_paths']
                ratio = (straight_paths / total_paths) if total_paths > 0 else 0.0

            # ✅ 接受概率策略（指数级拒绝）
            env_straight_ratio = n_straight / N
            
            if total_paths < 100:
                prob_accept = 1.0
            elif ratio < TARGET:
                prob_accept = 1.0 if env_straight_ratio <= TARGET else 0.6
            else:
                excess = (ratio - TARGET) / TARGET
                if n_straight == 0:
                    prob_accept = 1.0
                elif env_straight_ratio <= TARGET:
                    prob_accept = 0.9
                else:
                    # 指数级拒绝
                    k = 10
                    prob_accept = 0.01 * np.exp(-k * excess * env_straight_ratio)
                    prob_accept = max(prob_accept, 1e-6)

            if random.random() > prob_accept:
                safe_disconnect()
                try:
                    env.close()
                except:
                    pass
                with shared_stats['lock']:
                    shared_stats['rejected_envs'] = shared_stats.get('rejected_envs', 0) + 1
                return {"ok": False, "rejected": True, "prob_accept": prob_accept}

            # ✅ 接受环境
            with shared_stats['lock']:
                shared_stats['total_paths'] += N
                shared_stats['straight_paths'] += n_straight
                shared_stats['accepted_envs'] = shared_stats.get('accepted_envs', 0) + 1

            out = {
                "ok": True,
                "straight_flags": [bool(is_straight_path(p)) for p in paths],
                "env_range": config["env_range"],
                "pose_range": env.pose_range.tolist(),
                "start": [np_to_python(s) for s in starts],
                "goal": [np_to_python(g) for g in goals],
                "paths": [np_to_python(np.array(p)) for p in paths],
                "obstacles": np_to_python(obstacles),
                "num_obstacles": len(obstacles),
                "prob_accept": prob_accept,
            }

            try:
                env.close()
            except:
                pass
            safe_disconnect()
            return out

        # ✅ 失败，清理并重试
        try:
            env.close()
        except:
            pass

    # 所有重试失败
    safe_disconnect()
    with shared_stats['lock']:
        shared_stats['rejected_envs'] = shared_stats.get('rejected_envs', 0) + 1
    return {"ok": False, "error": "Failed to generate valid environment"}

# ----------------- 分块保存 -----------------
def save_chunk(chunk_id, chunk_data, save_dir):
    path = join(save_dir, f"chunk_{chunk_id}.json")
    with open(path, "w") as f:
        json.dump(chunk_data, f, indent=2, default=np_to_python)

def merge_chunks(save_dir, final_path):
    chunk_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("chunk_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    merged = []
    for cf in chunk_files:
        path = join(save_dir, cf)
        with open(path, "r") as f:
            merged.extend(json.load(f))
        os.remove(path)
    
    with open(final_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n✅ 合并完成 → {final_path}，共 {len(merged)} 个环境")

# ----------------- 数据集生成 -----------------

def load_existing_progress(data_dir):
    """
    扫描已有的 chunk 文件，统计进度
    
    Returns:
        dict: {
            'produced_envs': int,
            'total_paths': int,
            'straight_paths': int,
            'next_chunk_id': int,
            'existing_chunks': list
        }
    """
    if not os.path.exists(data_dir):
        return {
            'produced_envs': 0,
            'total_paths': 0,
            'straight_paths': 0,
            'next_chunk_id': 0,
            'existing_chunks': []
        }
    
    chunk_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("chunk_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    
    produced_envs = 0
    total_paths = 0
    straight_paths = 0
    
    for cf in chunk_files:
        path = join(data_dir, cf)
        with open(path, "r") as f:
            chunk_data = json.load(f)
        
        for env in chunk_data:
            produced_envs += 1
            n_paths = len(env.get('paths', []))
            n_straight = sum(env.get('straight_flags', []))
            total_paths += n_paths
            straight_paths += n_straight
    
    next_chunk_id = len(chunk_files)
    
    return {
        'produced_envs': produced_envs,
        'total_paths': total_paths,
        'straight_paths': straight_paths,
        'next_chunk_id': next_chunk_id,
        'existing_chunks': chunk_files
    }

# ✅ 修改：支持断点续传的数据集生成
def generate_env_dataset_parallel(config, timeout_per_env=30, resume=True):
    """
    Args:
        resume: 是否启用断点续传（默认 True）
    """
    env_type = config["env_type"]
    sizes = {
        "train": config["train_env_size"],
        "val": config["val_env_size"],
        "test": config["test_env_size"],
    }

    num_workers = min(cpu_count(), config.get("num_workers", cpu_count()))
    print(f"🧩 使用 {num_workers} 个进程")

    TARGET = config.get("target_straight_ratio", 0.2)
    print(f"🎯 目标直线路径占比: {TARGET:.1%}")

    for mode in ["train", "val", "test"]:
        target_envs = sizes[mode]
        data_dir = join("data", env_type, mode)
        os.makedirs(data_dir, exist_ok=True)
        
        # ✅ 加载已有进度
        progress = load_existing_progress(data_dir) if resume else {
            'produced_envs': 0,
            'total_paths': 0,
            'straight_paths': 0,
            'next_chunk_id': 0,
            'existing_chunks': []
        }
        
        produced_envs = progress['produced_envs']
        chunk_id = progress['next_chunk_id']
        
        print(f"\n{'='*60}")
        print(f"🚀 开始生成 {mode.upper()} 数据集")
        print(f"   目标环境数: {target_envs}")
        if progress['produced_envs'] > 0:
            print(f"   ✅ 已完成: {progress['produced_envs']} 个环境")
            print(f"   📊 已有数据: {progress['total_paths']} 条路径, "
                  f"直线率 {progress['straight_paths']/progress['total_paths']:.2%}")
            print(f"   🔄 从第 {progress['produced_envs']+1} 个环境继续")
        print(f"   每环境路径数: {config['num_samples_per_env']}")
        print(f"   障碍物数量: {config['num_boxes_range']}")
        print(f"   障碍物尺寸: {config['box_size_range']}")
        print(f"   超时设置: {timeout_per_env}s/环境")
        print(f"{'='*60}")
        
        # 如果已经完成，跳过
        if produced_envs >= target_envs:
            print(f"✅ {mode.upper()} 数据集已完成，跳过")
            continue
        
        remaining_envs = target_envs - produced_envs
        buffer = []
        
        manager = Manager()
        shared_stats = manager.dict()
        # ✅ 从已有进度初始化
        shared_stats['total_paths'] = progress['total_paths']
        shared_stats['straight_paths'] = progress['straight_paths']
        shared_stats['accepted_envs'] = progress['produced_envs']
        shared_stats['rejected_envs'] = 0
        shared_stats['timeout_envs'] = 0
        shared_stats['lock'] = manager.Lock()

        # ✅ 进度条从已完成数量开始
        pbar = tqdm(
            initial=produced_envs,
            total=target_envs,
            desc=f"📊 {mode.upper()}"
        )
        
        task_num = remaining_envs * 20
        start_time = time.time()

        with Pool(processes=num_workers) as pool:
            for env_dict in pool.imap_unordered(
                generate_single_env,
                ((i, config, shared_stats, TARGET, timeout_per_env) for i in range(task_num))
            ):
                if env_dict.get("timeout", False):
                    with shared_stats['lock']:
                        shared_stats['timeout_envs'] = shared_stats.get('timeout_envs', 0) + 1
                    continue
                
                if not env_dict.get("ok", False):
                    continue
                
                buffer.append(env_dict)
                produced_envs += 1
                pbar.update(1)
                
                current_ratio = (shared_stats['straight_paths'] / shared_stats['total_paths']) \
                    if shared_stats['total_paths'] > 0 else 0
                total_attempts = shared_stats['accepted_envs'] + shared_stats['rejected_envs']
                accept_rate = (shared_stats['accepted_envs'] / total_attempts) \
                    if total_attempts > 0 else 0
                
                pbar.set_postfix({
                    'straight': f"{current_ratio:.3f}",
                    'target': f"{TARGET:.3f}",
                    'accept%': f"{accept_rate*100:.1f}"
                })

                if len(buffer) >= 10:
                    save_chunk(chunk_id, buffer, data_dir)
                    buffer.clear()
                    chunk_id += 1
                
                if produced_envs >= target_envs:
                    pool.terminate()
                    break

        pbar.close()
        
        # 保存剩余数据
        if buffer:
            save_chunk(chunk_id, buffer, data_dir)

        # 合并所有分块
        final_file = join(data_dir, "envs.json")
        merge_chunks(data_dir, final_file)

        # 统计信息
        elapsed = time.time() - start_time
        final_ratio = shared_stats['straight_paths'] / shared_stats['total_paths'] \
            if shared_stats['total_paths'] > 0 else 0
        total_attempts = shared_stats['accepted_envs'] + shared_stats['rejected_envs']
        final_accept_rate = shared_stats['accepted_envs'] / total_attempts \
            if total_attempts > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ {mode.upper()} 数据集生成完成！")
        print(f"{'='*60}")
        print(f"⏱️  本次耗时: {elapsed/60:.1f} 分钟")
        print(f"📊 路径统计:")
        print(f"   - 总环境数: {produced_envs} 个")
        print(f"   - 总路径数: {shared_stats['total_paths']} 条")
        print(f"   - 直线路径: {shared_stats['straight_paths']} 条")
        print(f"   - 直线占比: {final_ratio:.2%} (目标: {TARGET:.2%})")
        print(f"📈 本次生成统计:")
        print(f"   - 接受: {shared_stats['accepted_envs'] - progress['produced_envs']} 次")
        print(f"   - 拒绝: {shared_stats['rejected_envs']} 次")
        print(f"   - 超时: {shared_stats['timeout_envs']} 次")
        print(f"   - 接受率: {final_accept_rate:.2%}")
        print(f"{'='*60}\n")

# ----------------- 主函数 -----------------
if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    config = {
        "env_type": "liche",
        "train_env_size": 5000,
        "val_env_size": 500,
        "test_env_size": 100,
        "num_samples_per_env": 5,
        "batch_size": 200,
        "GUI": False,
        "num_workers": 10,
        "target_straight_ratio": 0.3,
        "env_range": [[-7, 7], [-7, 7], [-2, 5]],
        "box_size_range": [0.5, 2.5],
        "num_boxes_range": [6, 20],
    }

    print("\n" + "="*60)
    print("⚙️  数据集生成配置")
    print("="*60)
    print(f"环境类型: {config['env_type']}")
    print(f"数据集规模: Train={config['train_env_size']}, "
          f"Val={config['val_env_size']}, Test={config['test_env_size']}")
    
    workspace_vol = np.prod([r[1]-r[0] for r in config['env_range']])
    avg_num_obs = np.mean(config['num_boxes_range'])
    avg_size = np.mean(config['box_size_range'])
    obs_vol = avg_num_obs * (avg_size ** 3)
    obs_ratio = obs_vol / workspace_vol
    
    print(f"工作空间体积: {workspace_vol:.0f} m³")
    print(f"障碍物配置: {config['num_boxes_range']} 个, 尺寸 {config['box_size_range']} m")
    print(f"预计障碍物占比: {obs_ratio:.2%}")
    print(f"目标直线路径占比: {config['target_straight_ratio']:.1%}")
    print(f"直线判定阈值: 1.05 (路径长度/直线距离)")
    print(f"规划参数: iter_max=500, batch_size=150")
    print(f"🔄 断点续传: 已启用")
    print("="*60 + "\n")

    # ✅ 启用断点续传（默认）
    generate_env_dataset_parallel(config, timeout_per_env=50, resume=True)
    
    print("\n🎉 全部完成！")