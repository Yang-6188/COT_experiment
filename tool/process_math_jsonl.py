import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3], [1, 4, 9], 'b-o', linewidth=2, markersize=10)
ax.set_xlabel('Token 位置', fontsize=14)
ax.set_ylabel('熵值', fontsize=14)
ax.set_title('中文显示测试 - 熵变化率分析', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(['测试数据'], fontsize=12)

plt.tight_layout()
plt.savefig('test_chinese.png', dpi=150, bbox_inches='tight')
print("✓ 测试图片已保存: test_chinese.png")
print("  请打开图片检查中文是否正常显示")
plt.close()
