# Import necessary libraries
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

def plot_class_accuracy(df_orig, df_dda, df_baseline, name):

    """
    Plot the class-wise accuracy for original, our augmented, and baseline datasets, 
    along with the proportion of each class in the original dataset.

    Args:
        df_orig: DataFrame containing the original predictions.
        df_dda: DataFrame containing the predictions of the declarative data augmentation (dda) approach.
        df_baseline: DataFrame containing the baseline predictions.
        name (str): Name or title to be used in the plot.

    This function calculates the accuracy for each class in the original, dda, and baseline predictions.
    It also calculates the proportion of each class in the original dataset.
    The results are plotted as grouped bar charts.
    """

    # Extract true labels and predictions from each approach
    y_true_orig = df_orig['ground_truth']
    y_pred_orig = df_orig['predicted']

    y_true_dda = df_dda['ground_truth']
    y_pred_dda = df_dda['predicted']

    y_true_baseline = df_baseline['ground_truth']
    y_pred_baseline = df_baseline['predicted']

    # Initialize dictionaries to store accuracy and class proportion for each class
    unique_classes = sorted(set(y_true_orig.unique()))
    class_accuracy_orig = {}
    class_accuracy_dda = {}
    class_accuracy_baseline = {}
    class_proportion = {}

    total_count_orig = len(y_true_orig)

    # Calculate accuracy and class proportion for each class
    for cls in unique_classes:
        mask_orig = y_true_orig == cls
        mask_dda = y_true_dda == cls
        mask_baseline = y_true_baseline == cls

        # Calculate class-specific accuracy for each dataset
        class_accuracy_orig[cls] = accuracy_score(y_true_orig[mask_orig], y_pred_orig[mask_orig]) * 100
        class_accuracy_dda[cls] = accuracy_score(y_true_dda[mask_dda], y_pred_dda[mask_dda]) * 100
        class_accuracy_baseline[cls] = accuracy_score(y_true_baseline[mask_baseline], y_pred_baseline[mask_baseline]) * 100

        # Calculate the proportion of the class in the original dataset
        class_proportion[cls] = (mask_orig.sum() / total_count_orig) * 100

    # Create a DataFrame for plotting with calculated accuracies and class proportions
    accuracy_df = pd.DataFrame({
        'Class': unique_classes,
        'Original Accuracy': [class_accuracy_orig[cls] for cls in unique_classes],
        'Baseline Accuracy': [class_accuracy_baseline[cls] for cls in unique_classes],
        'DDA Accuracy': [class_accuracy_dda[cls] for cls in unique_classes],
        'Class Proportion': [class_proportion[cls] for cls in unique_classes]
    })
    
    # Sort DataFrame by Class Proportion in descending order for better visualization
    accuracy_df.sort_values(by='Class Proportion', ascending=False, inplace=True)

    # Plot the class-wise accuracies and proportions
    fig, ax1 = plt.subplots(figsize=(20, 10))

    bar_width = 0.2
    x = range(len(accuracy_df))

    # Plot accuracy bars for Original, Baseline, and DDA datasets
    ax1.bar(x, accuracy_df['Original Accuracy'], width=bar_width, label='Original Accuracy', color='skyblue', edgecolor='gray', align='center')
    ax1.bar([p + bar_width for p in x], accuracy_df['Baseline Accuracy'], width=bar_width, label='Baseline Accuracy', color='lightcoral', edgecolor='gray', align='center')
    ax1.bar([p + 2 * bar_width for p in x], accuracy_df['DDA Accuracy'], width=bar_width, label='DDA Accuracy', color='lightgreen', edgecolor='gray', align='center')

    # Plot class proportion bars
    ax1.bar([p + 3 * bar_width for p in x], accuracy_df['Class Proportion'], width=bar_width, label='Class Proportion', color='gray', alpha=0.6, edgecolor='gray', align='center')

    # Set plot labels and title
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Value')
    plt.title(f'Class-wise Accuracy and Proportion for {name}')
    ax1.set_xticks([p + 1.5 * bar_width for p in x])
    ax1.set_xticklabels(accuracy_df['Class'])
    ax1.legend(loc='upper left')

    # # Annotate the bars with accuracy values
    # for i, acc in enumerate(accuracy_df['Original Accuracy']):
    #     ax1.text(i, acc, f'{acc:.1f}', ha='center', va='bottom')
    # for i, acc in enumerate(accuracy_df['Baseline Accuracy']):
    #     ax1.text(i + bar_width, acc, f'{acc:.1f}', ha='center', va='bottom')
    # for i, acc in enumerate(accuracy_df['DDA Accuracy']):
    #     ax1.text(i + 2 * bar_width, acc, f'{acc:.1f}', ha='center', va='bottom')

    # # Annotate the bars with proportion values
    # for i, prop in enumerate(accuracy_df['Class Proportion']):
    #     ax1.text(i + 3 * bar_width, prop, f'{prop:.1f}', ha='center', va='bottom', color='black')

    # Set plot limits
    ax1.set_xlim(-0.5, len(accuracy_df))

    plt.tight_layout()
    plt.show()

def map_categorization(df_base, df_adjust):

    """
    Align the 'ground_truth' and 'predicted' columns in the df_adjust DataFrame 
    based on the value counts in the df_base DataFrame.

    This function is useful when the categories in the adjusted DataFrame (df_adjust) 
    do not match those in the base DataFrame (df_base), possibly due to different 
    preprocessing or sampling methods.

    Args:
        df_base: The base DataFrame.
        df_adjust: The DataFrame that will be adjusted.

    Returns:
        df_adjust: The modified df_adjust DataFrame with adjsuted categorization.
    """
    
    # Get the value counts for ground_truth columns
    base_gt_counts = df_base['ground_truth'].value_counts().sort_index()
    adjust_gt_counts = df_adjust['ground_truth'].value_counts().sort_index()

    # Sort the value counts to align the categories
    base_gt_sorted = base_gt_counts.sort_values(ascending=False)
    adjust_gt_sorted = adjust_gt_counts.sort_values(ascending=False)

    # Generate a mapping dictionary by pairing the sorted values
    mapping = {adjust_val: base_val for adjust_val, base_val in zip(adjust_gt_sorted.index, base_gt_sorted.index)}

    # Apply the mapping to ground_truth and predicted columns in df_adjust
    df_adjust['ground_truth'] = df_adjust['ground_truth'].map(mapping).fillna(df_adjust['ground_truth'])
    df_adjust['predicted'] = df_adjust['predicted'].map(mapping).fillna(df_adjust['predicted'])

    return df_adjust
