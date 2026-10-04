"""
============================================================
OncoScan V2 — Dataset Exploration & Analysis
============================================================
Run this FIRST on Kaggle before any training.
Purpose: Understand the data thoroughly before making architecture decisions.

Instructions:
    1. Create a new Kaggle notebook
    2. Add dataset: "andrewmvd/isic-2019" (Skin Lesion Images for Melanoma Classification)
    3. Enable GPU is NOT needed for this script (saves quota)
    4. Copy-paste this code and run cell by cell
============================================================
"""

# %%==================== CELL 1: SETUP ====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

print("✅ Libraries loaded successfully")

# %%==================== CELL 2: FIND DATASET PATHS ====================
# Auto-detect dataset paths (handles different Kaggle path structures)
# Kaggle can mount datasets at:
#   /kaggle/input/<slug>/
#   /kaggle/input/datasets/<owner>/<name>/

INPUT_DIR = '/kaggle/input'

# Recursively search for the GroundTruth CSV
dataset_dir = None
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if 'GroundTruth' in f and f.endswith('.csv'):
            dataset_dir = root
            break
    if dataset_dir:
        break

if dataset_dir is None:
    raise FileNotFoundError(
        "❌ Could not find ISIC 2019 dataset. Make sure you added "
        "'andrewmvd/isic-2019' as input to your notebook.\n"
        f"Contents of {INPUT_DIR}: {os.listdir(INPUT_DIR)}"
    )

print(f"📁 Dataset found at: {dataset_dir}")
print(f"\n📂 Contents:")
for item in sorted(os.listdir(dataset_dir)):
    item_path = os.path.join(dataset_dir, item)
    if os.path.isdir(item_path):
        # Check for nested folders
        sub_items = os.listdir(item_path)
        count = len(sub_items)
        print(f"   📁 {item}/ ({count} items)")
        # If nested, check one level deeper
        for sub in sub_items[:3]:
            sub_path = os.path.join(item_path, sub)
            if os.path.isdir(sub_path):
                deep_count = len(os.listdir(sub_path))
                print(f"      📁 {sub}/ ({deep_count} items)")
    else:
        size_mb = os.path.getsize(item_path) / (1024 * 1024)
        print(f"   📄 {item} ({size_mb:.1f} MB)")

# %%==================== CELL 3: LOCATE FILES ====================
# Find CSV files and image directory

gt_csv = None
meta_csv = None
image_dir = None

for f in os.listdir(dataset_dir):
    fpath = os.path.join(dataset_dir, f)
    if 'GroundTruth' in f and f.endswith('.csv'):
        gt_csv = fpath
    elif 'Metadata' in f and f.endswith('.csv'):
        meta_csv = fpath
    elif os.path.isdir(fpath) and 'Training_Input' in f:
        # Handle potential nesting: ISIC_2019_Training_Input/ISIC_2019_Training_Input/
        nested = os.path.join(fpath, f)
        if os.path.isdir(nested):
            image_dir = nested
        else:
            # Check if images are directly here
            sample_files = [x for x in os.listdir(fpath) if x.endswith('.jpg')]
            if sample_files:
                image_dir = fpath
            else:
                # Images might be in a subfolder
                for sub in os.listdir(fpath):
                    sub_path = os.path.join(fpath, sub)
                    if os.path.isdir(sub_path):
                        sub_files = [x for x in os.listdir(sub_path) if x.endswith('.jpg')]
                        if sub_files:
                            image_dir = sub_path
                            break

print(f"\n🔍 Located files:")
print(f"   Ground Truth CSV: {gt_csv}")
print(f"   Metadata CSV:     {meta_csv}")
print(f"   Image Directory:  {image_dir}")

if image_dir:
    num_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    print(f"   Total Images:     {num_images:,}")

# %%==================== CELL 4: LOAD GROUND TRUTH ====================
print("=" * 70)
print("📊 GROUND TRUTH ANALYSIS")
print("=" * 70)

df_gt = pd.read_csv(gt_csv)
print(f"\nShape: {df_gt.shape}")
print(f"Columns: {list(df_gt.columns)}")
print(f"\nFirst 5 rows:")
print(df_gt.head())

# Class columns (all except 'image')
class_cols = [c for c in df_gt.columns if c != 'image']
print(f"\n🏷️ Classes found: {class_cols}")
print(f"   Total: {len(class_cols)} classes")

# Convert one-hot to single label
df_gt['label'] = df_gt[class_cols].idxmax(axis=1)
df_gt['label_idx'] = df_gt[class_cols].values.argmax(axis=1)

# Verify one-hot encoding (each row should sum to 1)
row_sums = df_gt[class_cols].sum(axis=1)
print(f"\n✅ One-hot verification:")
print(f"   Rows summing to 1: {(row_sums == 1).sum():,}")
print(f"   Rows summing to 0: {(row_sums == 0).sum():,} (if any, these are problematic)")
print(f"   Rows summing to >1: {(row_sums > 1).sum():,} (if any, multi-label)")

# %%==================== CELL 5: CLASS DISTRIBUTION ====================
print("\n" + "=" * 70)
print("📊 CLASS DISTRIBUTION (THE MOST IMPORTANT ANALYSIS)")
print("=" * 70)

class_counts = df_gt['label'].value_counts()
total = len(df_gt)

print(f"\n{'Class':<8} {'Full Name':<35} {'Count':>7} {'Percentage':>10} {'Ratio':>8}")
print("-" * 75)

class_full_names = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus',
    'BCC': 'Basal Cell Carcinoma',
    'AK': 'Actinic Keratosis',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular Lesion',
    'SCC': 'Squamous Cell Carcinoma',
    'UNK': 'None of the Above (Unknown)'
}

# Malignant vs Benign grouping
malignant_classes = {'MEL', 'BCC', 'AK', 'SCC'}
benign_classes = {'NV', 'BKL', 'DF', 'VASC'}

max_count = class_counts.max()
for cls in class_counts.index:
    count = class_counts[cls]
    pct = count / total * 100
    ratio = max_count / count
    full_name = class_full_names.get(cls, cls)
    nature = "🔴 Malignant" if cls in malignant_classes else ("🟢 Benign" if cls in benign_classes else "⚪ Unknown")
    print(f"{cls:<8} {full_name:<35} {count:>7,} {pct:>9.1f}% {ratio:>7.1f}x  {nature}")

print(f"\n{'TOTAL':<44} {total:>7,}")

# Imbalance ratio
print(f"\n⚖️ Imbalance Analysis:")
print(f"   Largest class:  {class_counts.index[0]} ({class_counts.iloc[0]:,})")
print(f"   Smallest class: {class_counts.index[-1]} ({class_counts.iloc[-1]:,})")
print(f"   Imbalance ratio: {max_count / class_counts.min():.1f}:1")

if max_count / class_counts.min() > 10:
    print(f"   ⚠️  SEVERE imbalance detected! Must use class weighting / oversampling / focal loss.")
elif max_count / class_counts.min() > 5:
    print(f"   ⚠️  Moderate imbalance. Class weighting recommended.")
else:
    print(f"   ✅ Relatively balanced.")

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Bar chart
colors = ['#e74c3c' if cls in malignant_classes else '#2ecc71' if cls in benign_classes else '#95a5a6'
          for cls in class_counts.index]
bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Class Distribution (ISIC 2019)', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Class', fontsize=13)
axes[0].set_ylabel('Number of Samples', fontsize=13)
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Malignant'),
                   Patch(facecolor='#2ecc71', label='Benign'),
                   Patch(facecolor='#95a5a6', label='Unknown/OOD')]
axes[0].legend(handles=legend_elements, loc='upper right', fontsize=11)

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, pctdistance=0.85,
            wedgeprops=dict(width=0.5, edgecolor='white'))
axes[1].set_title('Class Proportions', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('/kaggle/working/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("📸 Saved: /kaggle/working/class_distribution.png")

# %%==================== CELL 6: BINARY GROUPING ANALYSIS ====================
print("\n" + "=" * 70)
print("📊 BINARY GROUPING: MALIGNANT vs BENIGN vs UNKNOWN")
print("=" * 70)

def get_group(label):
    if label in malignant_classes:
        return 'Malignant'
    elif label in benign_classes:
        return 'Benign'
    else:
        return 'Unknown/OOD'

df_gt['binary_group'] = df_gt['label'].apply(get_group)
group_counts = df_gt['binary_group'].value_counts()

print(f"\n{'Group':<15} {'Count':>7} {'Percentage':>10}")
print("-" * 35)
for group in group_counts.index:
    count = group_counts[group]
    pct = count / total * 100
    print(f"{group:<15} {count:>7,} {pct:>9.1f}%")

print(f"\n🔍 Key Insight:")
print(f"   If we keep UNK: 9 classes (multi-class + OOD safety net)")
print(f"   If we drop UNK: 8 classes (cleaner, but no OOD detection)")

fig, ax = plt.subplots(figsize=(8, 6))
group_colors = {'Malignant': '#e74c3c', 'Benign': '#2ecc71', 'Unknown/OOD': '#95a5a6'}
bars = ax.bar(group_counts.index, group_counts.values,
              color=[group_colors[g] for g in group_counts.index],
              edgecolor='white', linewidth=2)
for bar, count in zip(bars, group_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{count:,}\n({count/total*100:.1f}%)', ha='center', va='bottom',
            fontsize=12, fontweight='bold')
ax.set_title('Malignant vs Benign vs Unknown Distribution', fontsize=16, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/binary_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%==================== CELL 7: METADATA ANALYSIS ====================
print("\n" + "=" * 70)
print("📊 METADATA ANALYSIS")
print("=" * 70)

if meta_csv:
    df_meta = pd.read_csv(meta_csv)
    print(f"\nMetadata shape: {df_meta.shape}")
    print(f"Columns: {list(df_meta.columns)}")
    print(f"\nFirst 5 rows:")
    print(df_meta.head())

    # Merge with ground truth
    df = df_gt.merge(df_meta, on='image', how='left')
    print(f"\nMerged dataset shape: {df.shape}")

    # ---- AGE ANALYSIS ----
    print(f"\n{'─' * 40}")
    print(f"🎂 AGE DISTRIBUTION")
    print(f"{'─' * 40}")

    if 'age_approx' in df.columns:
        age_stats = df['age_approx'].describe()
        print(f"   Mean:   {age_stats['mean']:.1f}")
        print(f"   Median: {age_stats['50%']:.1f}")
        print(f"   Std:    {age_stats['std']:.1f}")
        print(f"   Min:    {age_stats['min']:.0f}")
        print(f"   Max:    {age_stats['max']:.0f}")
        print(f"   NaN:    {df['age_approx'].isna().sum()} ({df['age_approx'].isna().mean()*100:.1f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Overall age distribution
        axes[0].hist(df['age_approx'].dropna(), bins=30, color='#3498db',
                     edgecolor='white', alpha=0.8)
        axes[0].set_title('Age Distribution (All Samples)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Count')
        axes[0].axvline(df['age_approx'].median(), color='red', linestyle='--',
                        label=f'Median: {df["age_approx"].median():.0f}')
        axes[0].legend()

        # Age distribution per class
        for cls in class_counts.index:
            subset = df[df['label'] == cls]['age_approx'].dropna()
            if len(subset) > 0:
                axes[1].hist(subset, bins=20, alpha=0.5, label=f'{cls} (n={len(subset)})')
        axes[1].set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Count')
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        plt.savefig('/kaggle/working/age_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Age by class (box plot)
        fig, ax = plt.subplots(figsize=(14, 6))
        df_plot = df.dropna(subset=['age_approx'])
        order = class_counts.index.tolist()
        sns.boxplot(data=df_plot, x='label', y='age_approx', order=order, ax=ax,
                    palette='husl')
        ax.set_title('Age Distribution per Class', fontsize=16, fontweight='bold')
        ax.set_xlabel('Class', fontsize=13)
        ax.set_ylabel('Age', fontsize=13)
        plt.tight_layout()
        plt.savefig('/kaggle/working/age_per_class.png', dpi=150, bbox_inches='tight')
        plt.show()

    # ---- SEX ANALYSIS ----
    print(f"\n{'─' * 40}")
    print(f"👤 SEX DISTRIBUTION")
    print(f"{'─' * 40}")

    if 'sex' in df.columns:
        sex_counts = df['sex'].value_counts(dropna=False)
        print(f"\n   {sex_counts.to_string()}")
        print(f"   NaN: {df['sex'].isna().sum()} ({df['sex'].isna().mean()*100:.1f}%)")

        # Sex distribution per class
        fig, ax = plt.subplots(figsize=(14, 6))
        sex_class = pd.crosstab(df['label'], df['sex'])
        sex_class.loc[order].plot(kind='bar', ax=ax, edgecolor='white')
        ax.set_title('Sex Distribution per Class', fontsize=16, fontweight='bold')
        ax.set_xlabel('Class', fontsize=13)
        ax.set_ylabel('Count', fontsize=13)
        ax.legend(title='Sex')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('/kaggle/working/sex_per_class.png', dpi=150, bbox_inches='tight')
        plt.show()

    # ---- ANATOMICAL SITE ANALYSIS ----
    print(f"\n{'─' * 40}")
    print(f"🦵 ANATOMICAL SITE DISTRIBUTION")
    print(f"{'─' * 40}")

    if 'anatom_site_general' in df.columns:
        site_counts = df['anatom_site_general'].value_counts(dropna=False)
        print(f"\n{site_counts.to_string()}")
        print(f"\nNaN: {df['anatom_site_general'].isna().sum()} ({df['anatom_site_general'].isna().mean()*100:.1f}%)")

        fig, ax = plt.subplots(figsize=(14, 6))
        site_class = pd.crosstab(df['label'], df['anatom_site_general'])
        site_class.loc[order].plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        ax.set_title('Anatomical Site Distribution per Class', fontsize=16, fontweight='bold')
        ax.set_xlabel('Class', fontsize=13)
        ax.set_ylabel('Count', fontsize=13)
        ax.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('/kaggle/working/site_per_class.png', dpi=150, bbox_inches='tight')
        plt.show()

else:
    print("⚠️ Metadata CSV not found. Skipping metadata analysis.")
    df = df_gt.copy()

# %%==================== CELL 8: MISSING DATA SUMMARY ====================
print("\n" + "=" * 70)
print("📊 MISSING DATA SUMMARY")
print("=" * 70)

if meta_csv:
    missing = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    if len(missing_df) > 0:
        print(f"\n{missing_df.to_string()}")
        print(f"\n💡 Strategy for missing values:")
        print(f"   - age_approx: Fill with median age per class (or global median)")
        print(f"   - sex: Encode as male=1, female=0, unknown/NaN=0.5")
        print(f"   - anatom_site_general: One-hot encode (NaN as separate category)")
    else:
        print("   ✅ No missing data found!")

# %%==================== CELL 9: IMAGE QUALITY ANALYSIS ====================
print("\n" + "=" * 70)
print("📊 IMAGE QUALITY & SIZE ANALYSIS")
print("=" * 70)

if image_dir:
    # Sample images for analysis (don't scan all 25K)
    all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    sample_size = min(500, len(all_image_files))
    sample_files = np.random.choice(all_image_files, size=sample_size, replace=False)

    widths, heights, file_sizes = [], [], []
    channels_list = []

    print(f"   Analyzing {sample_size} random images...")
    for fname in sample_files:
        fpath = os.path.join(image_dir, fname)
        try:
            img = Image.open(fpath)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            file_sizes.append(os.path.getsize(fpath) / 1024)  # KB
            channels_list.append(len(img.getbands()))
        except Exception as e:
            pass

    print(f"\n   📐 Image Dimensions (from {len(widths)} samples):")
    print(f"      Width  — Min: {min(widths)}, Max: {max(widths)}, "
          f"Mean: {np.mean(widths):.0f}, Median: {np.median(widths):.0f}")
    print(f"      Height — Min: {min(heights)}, Max: {max(heights)}, "
          f"Mean: {np.mean(heights):.0f}, Median: {np.median(heights):.0f}")
    print(f"      Channels: {Counter(channels_list)}")

    print(f"\n   💾 File Sizes:")
    print(f"      Min:    {min(file_sizes):.1f} KB")
    print(f"      Max:    {max(file_sizes):.1f} KB")
    print(f"      Mean:   {np.mean(file_sizes):.1f} KB")
    print(f"      Median: {np.median(file_sizes):.1f} KB")

    # Unique aspect ratios
    aspects = [w / h for w, h in zip(widths, heights)]
    print(f"\n   📏 Aspect Ratios:")
    print(f"      Min: {min(aspects):.3f}, Max: {max(aspects):.3f}, "
          f"Mean: {np.mean(aspects):.3f}")
    ar_type = 'square-ish' if abs(np.mean(aspects) - 1) < 0.1 else 'rectangular'
    print(f"      Most images are {ar_type}")

    # Plot dimensions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(widths, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0].set_title('Image Widths', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Width (px)')

    axes[1].hist(heights, bins=30, color='#e74c3c', edgecolor='white', alpha=0.8)
    axes[1].set_title('Image Heights', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Height (px)')

    axes[2].scatter(widths, heights, alpha=0.3, s=10, c='#2ecc71')
    axes[2].set_title('Width vs Height', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Width (px)')
    axes[2].set_ylabel('Height (px)')
    axes[2].plot([0, max(widths)], [0, max(heights)], 'r--', alpha=0.5, label='1:1 ratio')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('/kaggle/working/image_dimensions.png', dpi=150, bbox_inches='tight')
    plt.show()

    # What this means for our input size
    print(f"\n   💡 Input Size Recommendation:")
    print(f"      Original images are ~{np.median(widths):.0f}x{np.median(heights):.0f}")
    print(f"      EfficientNet-B4 native: 380x380")
    print(f"      EfficientNet-B5 native: 456x456")
    print(f"      Resizing to 380x380 preserves most detail while fitting in GPU memory")

# %%==================== CELL 10: SAMPLE IMAGES PER CLASS ====================
print("\n" + "=" * 70)
print("🖼️ SAMPLE IMAGES FROM EACH CLASS")
print("=" * 70)

if image_dir:
    # Get classes including UNK
    all_classes = class_counts.index.tolist()
    n_classes = len(all_classes)
    n_samples_per_class = 4

    fig, axes = plt.subplots(n_classes, n_samples_per_class,
                              figsize=(n_samples_per_class * 4, n_classes * 3.5))

    for i, cls in enumerate(all_classes):
        cls_images = df_gt[df_gt['label'] == cls]['image'].values
        selected = np.random.choice(cls_images, size=min(n_samples_per_class, len(cls_images)),
                                     replace=False)

        nature = "🔴" if cls in malignant_classes else ("🟢" if cls in benign_classes else "⚪")
        full_name = class_full_names.get(cls, cls)

        for j in range(n_samples_per_class):
            ax = axes[i][j] if n_classes > 1 else axes[j]
            if j < len(selected):
                img_path = os.path.join(image_dir, selected[j] + '.jpg')
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    ax.imshow(img)
                    if j == 0:
                        ax.set_ylabel(f'{nature} {cls}\n({full_name})',
                                       fontsize=10, fontweight='bold', rotation=0,
                                       labelpad=120, va='center')
            ax.axis('off')

    plt.suptitle('Sample Images from Each Class', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('/kaggle/working/sample_images.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("📸 Saved: /kaggle/working/sample_images.png")

# %%==================== CELL 11: CROSS-ANALYSIS ====================
print("\n" + "=" * 70)
print("📊 CROSS-ANALYSIS: CLASS x METADATA CORRELATIONS")
print("=" * 70)

if meta_csv and 'age_approx' in df.columns:
    # Heatmap: Mean age per class x sex
    print("\n📋 Mean Age by Class x Sex:")
    cross = df.groupby(['label', 'sex'])['age_approx'].mean().unstack()
    print(cross.round(1).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cross, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=1, linecolor='white')
    ax.set_title('Mean Age by Class x Sex', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/kaggle/working/age_sex_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%==================== CELL 12: UNK CLASS DEEP DIVE ====================
print("\n" + "=" * 70)
print("🔬 DEEP DIVE: UNK (None of the Above) CLASS")
print("=" * 70)

unk_count = class_counts.get('UNK', 0)
if unk_count > 0:
    print(f"\n   UNK samples: {unk_count:,} ({unk_count/total*100:.1f}% of dataset)")

    if meta_csv and 'age_approx' in df.columns:
        unk_data = df[df['label'] == 'UNK']
        non_unk_data = df[df['label'] != 'UNK']

        print(f"\n   Age comparison:")
        print(f"      UNK mean age:     {unk_data['age_approx'].mean():.1f}")
        print(f"      Non-UNK mean age: {non_unk_data['age_approx'].mean():.1f}")

        if 'sex' in df.columns:
            print(f"\n   Sex distribution in UNK:")
            print(f"      {unk_data['sex'].value_counts().to_string()}")

        if 'anatom_site_general' in df.columns:
            print(f"\n   Anatomical sites in UNK:")
            print(f"      {unk_data['anatom_site_general'].value_counts().to_string()}")

    print(f"\n   💡 UNK Class Analysis:")
    unk_size_label = 'Sufficient' if unk_count > 500 else 'Small (may need augmentation)'
    print(f"      - Size: {unk_count:,} samples -> {unk_size_label}")
    print(f"      - Purpose: Acts as out-of-distribution (OOD) safety class")
    print(f"      - Recommendation: KEEP IT -> prevents false-positive cancer diagnoses")
    print(f"      - In production: If model predicts UNK, show 'Unrecognized lesion - consult a doctor'")
else:
    print("   ⚠️ No UNK class found in this dataset version.")

# %%==================== CELL 13: FINAL SUMMARY & RECOMMENDATIONS ====================
print("\n" + "=" * 70)
print("📋 FINAL DATASET SUMMARY & RECOMMENDATIONS")
print("=" * 70)

mal_total = sum(class_counts.get(c, 0) for c in malignant_classes)
ben_total = sum(class_counts.get(c, 0) for c in benign_classes)
unk_total = class_counts.get('UNK', 0)

print(f"""
=====================================
  DATASET: ISIC 2019
  Total Images: {total:,}
  Classes: {len(class_cols)} ({', '.join(class_cols)})
  Malignant: {mal_total:,}
  Benign: {ben_total:,}
  Unknown/OOD: {unk_total:,}
  Imbalance Ratio: {max_count / class_counts.min():.1f}:1
=====================================

🎯 RECOMMENDATIONS FOR TRAINING:

1. CLASSES: Keep all 9 classes (including UNK for OOD detection)
   -> Group into: Malignant (4) | Benign (4) | Unknown (1)

2. CLASS IMBALANCE: Severe — must handle with:
   -> Focal Loss (gamma=2.0)
   -> Class weights inversely proportional to frequency
   -> Oversample minority classes (DF, VASC, SCC)

3. METADATA: Use age + sex (skip anatom_site for now)
   -> Fill NaN ages with class-wise median
   -> Encode sex: male=1, female=0, NaN=0.5

4. IMAGE SIZE: 380x380 (EfficientNet-B4 native)

5. AUGMENTATION: Heavy augmentation for minority classes
   -> HFlip, VFlip, Rotate, ShiftScaleRotate
   -> ColorJitter, CoarseDropout, GaussNoise
""")

print("✅ EDA COMPLETE! Review the plots above, then proceed to training.")
print("📁 All plots saved to /kaggle/working/")
