import matplotlib.pyplot as plt
import os
def loss_plot(args,loss):
    num = args.epoch
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'_loss.jpg'
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)

def metrics_plot(args, metric_type, *metrics):
    """绘制指标曲线
    Args:
        args: 配置参数
        metric_type: 指标类型（用于标题和标签）
        *metrics: 可变参数，传入多个指标列表（如 [iou_list, dice_list]）
    """
    plt.figure(figsize=(10, 5))
    x = range(len(metrics[0]))  #  epoch 数量（所有指标列表长度一致）
    
    # 根据指标类型定义标签
    if metric_type == 'iou&dice':
        names = ['IoU', 'Dice']
    elif metric_type == 'small_vessel':
        names = ['Small Vessel Recall', 'Small Vessel Precision']  # 2个标签，对应2个指标
    elif metric_type == 'hd':
        names = ['HD']  # 1个标签，对应1个指标
    else:
        names = [f'Metric_{i+1}' for i in range(len(metrics))]  # 兜底：自动生成标签
    
    # 绘制每个指标
    for i, l in enumerate(metrics):
        if i < len(names):
            plt.plot(x, l, label=names[i])
        else:
            plt.plot(x, l, label=f'Metric_{i+1}')
    
    # 图表配置
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title(f'{args.arch} {metric_type} Curve (Batch={args.batch_size}, Epoch={args.epoch})')
    plt.legend()
    plt.grid(True)
    # 保存路径
    save_dir = os.path.join(args.log_dir, args.arch, str(args.batch_size), str(args.dataset), str(args.epoch))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{metric_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()