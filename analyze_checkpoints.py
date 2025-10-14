"""
Checkpoint Analysis and Visualization
Generate summary report from evaluation results
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(results_file='evaluation_results.json'):
    """Load evaluation results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_summary_report(results):
    """Create a detailed summary report."""
    checkpoints = results['checkpoints']
    
    # Sort by epoch
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x['epoch'])
    
    print("=" * 80)
    print("STREAMSPLIT CHECKPOINT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nEvaluation Time: {results['evaluation_time']}")
    print(f"Device: {results['device']}")
    print(f"Config: {results['config']}")
    print(f"\nTotal Checkpoints: {len(checkpoints)}")
    
    # Find best checkpoint
    best_checkpoint = min(sorted_checkpoints, key=lambda x: x['train_loss'])
    worst_checkpoint = max(sorted_checkpoints, key=lambda x: x['train_loss'])
    final_checkpoint = sorted_checkpoints[-1]
    
    print("\n" + "-" * 80)
    print("KEY STATISTICS")
    print("-" * 80)
    
    print(f"\n🏆 BEST CHECKPOINT: {best_checkpoint['checkpoint']}")
    print(f"   Epoch: {best_checkpoint['epoch'] + 1}")
    print(f"   Train Loss: {best_checkpoint['train_loss']:.4f}")
    
    print(f"\n📊 FINAL CHECKPOINT: {final_checkpoint['checkpoint']}")
    print(f"   Epoch: {final_checkpoint['epoch'] + 1}")
    print(f"   Train Loss: {final_checkpoint['train_loss']:.4f}")
    
    print(f"\n⚠️  HIGHEST LOSS: {worst_checkpoint['checkpoint']}")
    print(f"   Epoch: {worst_checkpoint['epoch'] + 1}")
    print(f"   Train Loss: {worst_checkpoint['train_loss']:.4f}")
    
    # Calculate improvement
    first_checkpoint = sorted_checkpoints[0]
    improvement = ((first_checkpoint['train_loss'] - final_checkpoint['train_loss']) / 
                   first_checkpoint['train_loss'] * 100)
    
    print(f"\n📈 OVERALL IMPROVEMENT")
    print(f"   Initial Loss (Epoch {first_checkpoint['epoch'] + 1}): "
          f"{first_checkpoint['train_loss']:.4f}")
    print(f"   Final Loss (Epoch {final_checkpoint['epoch'] + 1}): "
          f"{final_checkpoint['train_loss']:.4f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    # Loss statistics
    losses = [cp['train_loss'] for cp in sorted_checkpoints]
    print(f"\n📉 LOSS STATISTICS")
    print(f"   Mean: {np.mean(losses):.4f}")
    print(f"   Std Dev: {np.std(losses):.4f}")
    print(f"   Min: {np.min(losses):.4f}")
    print(f"   Max: {np.max(losses):.4f}")
    
    # Detailed table
    print("\n" + "-" * 80)
    print("DETAILED CHECKPOINT TABLE")
    print("-" * 80)
    print(f"{'Checkpoint':<25} {'Epoch':<8} {'Train Loss':<12} {'Status':<15}")
    print("-" * 80)
    
    for cp in sorted_checkpoints:
        status = ""
        if cp == best_checkpoint:
            status = "🏆 BEST"
        elif cp == final_checkpoint:
            status = "📊 FINAL"
        elif cp == worst_checkpoint:
            status = "⚠️  HIGHEST"
        
        print(f"{cp['checkpoint']:<25} {cp['epoch']+1:<8} "
              f"{cp['train_loss']:<12.4f} {status:<15}")
    
    print("-" * 80)
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if best_checkpoint['epoch'] < 80:
        print("\n⚠️  Note: Best checkpoint was achieved at epoch "
              f"{best_checkpoint['epoch'] + 1}")
        print("   Consider using this checkpoint for inference rather than the final one.")
        print("   This suggests the model may have overfit in later epochs.")
    else:
        print("\n✓ Best checkpoint is from later training stages.")
        print("  The model continued to improve throughout training.")
    
    loss_variance = np.std(losses[-5:]) if len(losses) >= 5 else np.std(losses)
    if loss_variance < 0.1:
        print("\n✓ Training has converged (low variance in recent epochs).")
    else:
        print(f"\n⚠️  High variance in recent epochs ({loss_variance:.4f}).")
        print("   Model may benefit from additional training or learning rate reduction.")
    
    print("\n" + "=" * 80)
    
    return sorted_checkpoints


def create_visualization(checkpoints, output_file='checkpoint_analysis.png'):
    """Create visualization of training progress."""
    epochs = [cp['epoch'] + 1 for cp in checkpoints]
    losses = [cp['train_loss'] for cp in checkpoints]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss over epochs
    ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=8, alpha=0.7)
    ax1.axhline(y=min(losses), color='g', linestyle='--', 
                label=f'Best: {min(losses):.4f}', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Annotate best checkpoint
    best_idx = losses.index(min(losses))
    ax1.annotate(f'Best\n{min(losses):.4f}', 
                xy=(epochs[best_idx], losses[best_idx]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Loss improvement histogram
    if len(losses) > 1:
        improvements = [-100 * (losses[i] - losses[i-1]) / losses[i-1] 
                       for i in range(1, len(losses))]
        colors = ['g' if imp > 0 else 'r' for imp in improvements]
        ax2.bar(range(1, len(improvements) + 1), improvements, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Checkpoint Interval', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss Improvement (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Checkpoint-to-Checkpoint Improvement', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_file}")
    plt.close()


def export_markdown_report(checkpoints, output_file='CHECKPOINT_REPORT.md'):
    """Export results as markdown report."""
    best_checkpoint = min(checkpoints, key=lambda x: x['train_loss'])
    final_checkpoint = checkpoints[-1]
    
    with open(output_file, 'w') as f:
        f.write("# StreamSplit Checkpoint Evaluation Report\n\n")
        f.write(f"**Generated:** {Path.cwd().name}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Checkpoints Evaluated:** {len(checkpoints)}\n")
        f.write(f"- **Best Checkpoint:** `{best_checkpoint['checkpoint']}` "
                f"(Epoch {best_checkpoint['epoch'] + 1}, "
                f"Loss: {best_checkpoint['train_loss']:.4f})\n")
        f.write(f"- **Final Checkpoint:** `{final_checkpoint['checkpoint']}` "
                f"(Epoch {final_checkpoint['epoch'] + 1}, "
                f"Loss: {final_checkpoint['train_loss']:.4f})\n\n")
        
        f.write("## Training Progress\n\n")
        f.write("| Checkpoint | Epoch | Training Loss | Notes |\n")
        f.write("|------------|-------|---------------|-------|\n")
        
        for cp in checkpoints:
            notes = ""
            if cp == best_checkpoint:
                notes = "🏆 **Best**"
            elif cp == final_checkpoint:
                notes = "📊 Final"
            
            f.write(f"| {cp['checkpoint']} | {cp['epoch'] + 1} | "
                   f"{cp['train_loss']:.4f} | {notes} |\n")
        
        f.write("\n## Recommendations\n\n")
        f.write(f"For inference and deployment, use: **{best_checkpoint['checkpoint']}**\n\n")
        f.write("This checkpoint achieved the lowest training loss and represents "
                "the best model performance.\n")
    
    print(f"\n📄 Markdown report saved to: {output_file}")


def main():
    """Main analysis pipeline."""
    # Load results
    results = load_results()
    
    # Create summary report
    checkpoints = create_summary_report(results)
    
    # Create visualizations
    try:
        create_visualization(checkpoints)
    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")
    
    # Export markdown report
    export_markdown_report(checkpoints)
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
