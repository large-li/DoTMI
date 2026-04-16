import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_data_with_labels(features_file, labels_file):
    """
    加载特征向量和标签数据
    features_file: 特征向量文件路径
    labels_file: 标签文件路径 (1=成员, 0=非成员)
    """
    # 从txt文件加载特征向量（支持单列 baseline 或双列 DoTMI）
    features = []
    with open(features_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = [x.strip() for x in line.split(",") if x.strip() != ""]
                values = list(map(float, parts))
                features.append(values)
    features = np.array(features)
    
    # 从txt文件加载标签
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    labels = np.array(labels)
    
    return features, labels

def find_optimal_threshold(features, labels):
    """
    遍历所有可能的阈值组合，找到最优阈值theta*使AUC最大
    """
    hamming_distances = features[:, 0]
    ppl_values = features[:, 1]
    
    # 生成候选阈值
    hamming_thresholds = np.linspace(hamming_distances.min(), hamming_distances.max(), 50)
    ppl_thresholds = np.linspace(ppl_values.min(), ppl_values.max(), 50)
    
    best_auc = 0
    best_theta = None
    best_predictions = None
    
    print("正在搜索最优阈值...")
    
    for theta_d in hamming_thresholds:
        for theta_ppl in ppl_thresholds:
            # 根据阈值进行预测
            predictions = ((hamming_distances <= theta_d) & (ppl_values <= theta_ppl)).astype(int)
            
            # 计算AUC
            try:
                auc = roc_auc_score(labels, predictions)
                if auc > best_auc:
                    best_auc = auc
                    best_theta = (theta_d, theta_ppl)
                    best_predictions = predictions
            except ValueError:
                # 如果所有预测都相同，跳过
                continue
    
    return best_theta, best_auc, best_predictions


def evaluate_single_score(features, labels, lower_score_is_member=True):
    """
    适用于 baseline（单列分数）：
    - lower_score_is_member=True: 分数越小越偏成员（如PPL、Min-K NLL）
    """
    scores = features[:, 0]
    eval_scores = -scores if lower_score_is_member else scores
    auc = roc_auc_score(labels, eval_scores)
    return auc, eval_scores

def calculate_tpr_at_fpr(labels, predictions, fpr_values=[0.01, 0.05, 0.1]):
    """
    计算在不同FPR下的TPR值
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    
    tpr_at_fpr = {}
    for target_fpr in fpr_values:
        # 找到最接近目标FPR的索引
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_fpr[target_fpr] = tpr[idx]
    
    return tpr_at_fpr

def evaluate_membership_detection(features_file, labels_file):
    """
    评估成员检测效果
    """
    # 加载数据
    features, labels = load_data_with_labels(features_file, labels_file)
    
    print(f"数据集大小: {len(features)} 本书")
    print(f"成员数据: {np.sum(labels)} 本")
    print(f"非成员数据: {len(labels) - np.sum(labels)} 本")
    
    if features.ndim == 1 or (features.ndim == 2 and features.shape[1] == 1):
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        auc_score, eval_scores = evaluate_single_score(features, labels, lower_score_is_member=True)
        fpr, tpr, _ = roc_curve(labels, eval_scores)
        tpr_at_fpr = {}
        for target_fpr in [0.01, 0.05, 0.1]:
            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_fpr[target_fpr] = tpr[idx]
        print(f"\n单列 baseline 模式 AUC: {auc_score:.4f}")
        print("TPR@FPR:")
        for fp, tp in tpr_at_fpr.items():
            print(f"  FPR={fp}: TPR={tp:.4f}")
        return None, auc_score, tpr_at_fpr
    else:
        # 找到最优阈值
        best_theta, best_auc, best_predictions = find_optimal_threshold(features, labels)

        print(f"\n最优阈值 theta* = (θ_d={best_theta[0]:.4f}, θ_ppl={best_theta[1]:.4f})")
        print(f"最优AUC分数: {best_auc:.4f}")

        # 计算TPR@FPR
        tpr_at_fpr = calculate_tpr_at_fpr(labels, best_predictions)

        print(f"\nTPR@FPR 结果:")
        for fpr, tpr in tpr_at_fpr.items():
            print(f"TPR@FPR={fpr}: {tpr:.4f}")

        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, best_predictions)
        print(f"\n混淆矩阵:")
        print(f"真负例(TN): {cm[0,0]}, 假正例(FP): {cm[0,1]}")
        print(f"假负例(FN): {cm[1,0]}, 真正例(TP): {cm[1,1]}")

        # 计算精确率和召回率
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1_score:.4f}")
        return best_theta, best_auc, tpr_at_fpr

def plot_roc_curve(features_file, labels_file):
    """
    绘制ROC曲线
    """
    features, labels = load_data_with_labels(features_file, labels_file)
    if features.ndim == 1 or (features.ndim == 2 and features.shape[1] == 1):
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        auc, eval_scores = evaluate_single_score(features, labels, lower_score_is_member=True)
        fpr, tpr, _ = roc_curve(labels, eval_scores)
    else:
        _, _, best_predictions = find_optimal_threshold(features, labels)
        fpr, tpr, _ = roc_curve(labels, best_predictions)
        auc = roc_auc_score(labels, best_predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Membership Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_methods(methods=['no_filter', 'uniqueness', 'density']):
    """
    对比不同筛选方法的AUC性能
    methods: 要对比的方法列表
    """
    print("="*60)
    print("方法对比评估")
    print("="*60)
    
    comparison_results = {}
    
    for method in methods:
        feature_file = f"gutenberg_features_{method}.txt"
        label_file = f"gutenberg_labels_{method}.txt"
        
        try:
            print(f"\n评估方法: {method}")
            best_theta, auc_score, tpr_at_fpr = evaluate_membership_detection(
                feature_file, label_file
            )
            
            comparison_results[method] = {
                "auc": auc_score,
                "theta": best_theta,
                "tpr_at_fpr": tpr_at_fpr
            }
            
        except FileNotFoundError:
            print(f"  警告: {feature_file} 或 {label_file} 不存在，跳过")
            continue
        except Exception as e:
            print(f"  评估方法 {method} 时出错: {e}")
            continue
    
    # 打印对比结果
    print("\n" + "="*60)
    print("对比结果总结")
    print("="*60)
    print(f"{'方法':<20} {'AUC':<10} {'θ_d':<12} {'θ_ppl':<12}")
    print("-"*60)
    
    for method, res in comparison_results.items():
        print(f"{method:<20} {res['auc']:<10.4f} {res['theta'][0]:<12.4f} {res['theta'][1]:<12.4f}")
    
    # 保存对比结果
    with open("auc_comparison.txt", "w") as f:
        f.write("方法AUC对比结果\n")
        f.write("="*60 + "\n\n")
        for method, res in comparison_results.items():
            f.write(f"方法: {method}\n")
            f.write(f"  AUC: {res['auc']:.4f}\n")
            f.write(f"  最优阈值: θ_d={res['theta'][0]:.4f}, θ_ppl={res['theta'][1]:.4f}\n")
            f.write(f"  TPR@FPR:\n")
            for fpr, tpr in res['tpr_at_fpr'].items():
                f.write(f"    FPR={fpr}: TPR={tpr:.4f}\n")
            f.write("\n")
    
    print("\n对比结果已保存: auc_comparison.txt")
    
    return comparison_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # 对比模式
        methods = sys.argv[2:] if len(sys.argv) > 2 else ['no_filter', 'uniqueness', 'density']
        compare_methods(methods)
    else:
        # 单一评估模式
        feature_file = sys.argv[1] if len(sys.argv) > 1 else "gutenberg_features.txt"
        label_file = sys.argv[2] if len(sys.argv) > 2 else "gutenberg_labels.txt"
        
        # 评估成员检测效果
        best_theta, auc_score, tpr_at_fpr = evaluate_membership_detection(
            feature_file, label_file
        )
        
        # 绘制ROC曲线
        plot_roc_curve(feature_file, label_file)
