import math
import io
from typing import List
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from maestro.constants import label_to_source

def _save_figures_as_collage_matplotlib(figures: List[plt.Figure], filename: str = "collage.png", cols: int = 2):
    """
    Saves a list of Matplotlib figures as a single collage image.

    Parameters:
    - figures: List of Matplotlib figure objects to include in the collage.
    - filename: Name of the output collage image file.
    - cols: Number of columns in the collage grid.
    """
    # Determine the number of rows needed based on columns
    rows = math.ceil(len(figures) / cols)
    fig_collage, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    
    # Flatten axes array for easy iteration
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for ax in axes:
        ax.axis('off')  # Hide all axes initially
    
    for i, individual_fig in enumerate(figures):
        if i >= len(axes):
            break  # Prevent index error if figures exceed axes
        # Save individual figure to a buffer
        buf = io.BytesIO()
        individual_fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
    
        # Display the image on the collage axis
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')  # Hide axes ticks
    
        # Close the individual figure to free memory
        plt.close(individual_fig)
    
    plt.tight_layout()
    fig_collage.savefig(filename)
    plt.close(fig_collage)  # Close the collage figure to avoid display

class GTZANLogger():
    def __init__(self, top_k=None, loss_by_source=False):
        self.entries = []
        self.epoch = 0
        self.top_k = top_k
        
        self.current_entry = GTZANLogEntry(self.epoch, top_k=top_k)

        self.has_accuracy = top_k is not None
        self.loss_by_source = loss_by_source

    def step(self):
        self.entries.append(self.current_entry)
        self.epoch += 1
        self.current_entry = GTZANLogEntry(self.epoch, top_k=self.top_k)
    
    def train_loss(self, loss, source_losses=None):
        self.current_entry.train_loss = loss
        self.current_entry.source_train_losses = source_losses
        self.current_entry.print_train_loss()
    
    def val_loss(self, loss, source_losses=None):
        self.current_entry.val_loss = loss
        self.current_entry.source_val_losses = source_losses
        self.current_entry.print_val_loss()
    
    def accuracy(self, acc, source_accs=None):
        self.current_entry.accuracy = acc
        self.current_entry.source_accuracies = source_accs
        self.current_entry.print_accuracy()

    def _get_figures(self, title: str) -> List[plt.Figure]:
        """
        Generates graphs for training and validation metrics, including per-source plots.

        Parameters:
        - title: Base title for the graphs.

        Returns:
        - List of Matplotlib figure objects.
        """

        k = self.top_k[0] if isinstance(self.top_k, list) else self.top_k

        figures: List[plt.Figure] = []
        epochs = [entry.epoch for entry in self.entries]
        train_losses = [entry.train_loss for entry in self.entries]
        val_losses = [entry.val_loss for entry in self.entries]
        
        # Handle multiple top_k values by selecting the first one
        if self.has_accuracy:
            accuracies = [entry.accuracy for entry in self.entries]
            if isinstance(self.top_k, list):
                accuracies = [acc[0] for acc in accuracies]
        else:
            accuracies = None

        # === Overall Analysis ===
        # 1. Overall Train Loss, Validation Loss, and Accuracy
        fig_overall, ax1 = plt.subplots()
        ax1.plot(epochs, train_losses, label="Train Loss", color="blue")
        ax1.plot(epochs, val_losses, label="Validation Loss", color="orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left")

        # Headroom for Loss
        headroom_multiplier = 1.3  # 30% headroom
        max_loss = max(max(train_losses, default=0), max(val_losses, default=0))
        ax1.set_ylim(0, max_loss * headroom_multiplier)
        
        if self.has_accuracy:


            ax2 = ax1.twinx()
            ax2.plot(epochs, accuracies, label=f"Accuracy @{k}", color="green")
            ax2.set_ylabel(f"Accuracy @{k}")
            ax2.set_ylim(0, 1.0 * headroom_multiplier)  # Assuming accuracy ranges between 0 and 1
            ax2.legend(loc="upper right")

        # Update the title
        overall_title = f"{title} - Overall: Train vs Validation Loss"
        if self.has_accuracy:
            overall_title += f" with Accuracy @{k}"
        fig_overall.suptitle(overall_title)
        figures.append(fig_overall)

        # 2. Train Loss per Source
        if self.loss_by_source and self.entries:
            sources = self.entries[0].source_train_losses.keys()
            
            # Train Loss per Source
            fig_train_sources, ax_train_sources = plt.subplots()
            for source in sources:
                source_train_losses = [
                    entry.source_train_losses.get(source, np.nan) for entry in self.entries
                ]
                ax_train_sources.plot(epochs, source_train_losses, label=label_to_source(source))
            ax_train_sources.set_xlabel("Epoch")
            ax_train_sources.set_ylabel("Train Loss")
            ax_train_sources.legend(loc="upper left")
            
            # Headroom for Train Loss per Source
            max_train_source = max([max([loss for loss in entry.source_train_losses.values()], default=0) for entry in self.entries], default=0)
            ax_train_sources.set_ylim(0, max_train_source * headroom_multiplier)
            
            fig_train_sources.suptitle(f"{title} - Train Loss by Source")
            figures.append(fig_train_sources)
            
            # 3. Validation Loss per Source
            fig_val_sources, ax_val_sources = plt.subplots()
            for source in sources:
                source_val_losses = [
                    entry.source_val_losses.get(source, np.nan) for entry in self.entries
                ]
                ax_val_sources.plot(epochs, source_val_losses, label=label_to_source(source))
            ax_val_sources.set_xlabel("Epoch")
            ax_val_sources.set_ylabel("Validation Loss")
            ax_val_sources.legend(loc="upper left")
            
            # Headroom for Validation Loss per Source
            max_val_source = max([max([loss for loss in entry.source_val_losses.values()], default=0) for entry in self.entries], default=0)
            ax_val_sources.set_ylim(0, max_val_source * headroom_multiplier)
            
            fig_val_sources.suptitle(f"{title} - Validation Loss by Source")
            figures.append(fig_val_sources)
            
            # 4. Accuracy per Source
            if self.has_accuracy:

                fig_acc_sources, ax_acc_sources = plt.subplots()
                for source in sources:

                    source_acc = [
                        entry.source_accuracies.get(label_to_source(source), np.nan) for entry in self.entries
                    ]

                    if isinstance(self.top_k, list):
                        source_acc = [acc[0] for acc in source_acc]

                    ax_acc_sources.plot(epochs, source_acc, label=label_to_source(source))
                ax_acc_sources.set_xlabel("Epoch")
                ax_acc_sources.set_ylabel(f"Accuracy @{k}")
                ax_acc_sources.legend(loc="upper left")
                
                # Headroom for Accuracy per Source
                ax_acc_sources.set_ylim(0, 1.0 * headroom_multiplier)  # Assuming accuracy ranges between 0 and 1
                
                fig_acc_sources.suptitle(f"{title} - Accuracy @{k} by Source")
                figures.append(fig_acc_sources)
        
        # === Per-Source Analysis ===
        if self.loss_by_source and self.entries:
            for source in sources:
                # Combined Train Loss, Validation Loss, and Accuracy for each Source
                fig_source, ax1_source = plt.subplots()
                
                # Plot Train and Validation Loss
                source_train_losses = [
                    entry.source_train_losses.get(source, np.nan) for entry in self.entries
                ]
                source_val_losses = [
                    entry.source_val_losses.get(source, np.nan) for entry in self.entries
                ]
                ax1_source.plot(epochs, source_train_losses, label="Train Loss", color="blue")
                ax1_source.plot(epochs, source_val_losses, label="Validation Loss", color="orange")
                ax1_source.set_xlabel("Epoch")
                ax1_source.set_ylabel("Loss")
                ax1_source.legend(loc="upper left")
                
                # Headroom for Loss
                max_loss_source = max(max(source_train_losses, default=0), max(source_val_losses, default=0))
                ax1_source.set_ylim(0, max_loss_source * headroom_multiplier)
                
                if self.has_accuracy:
                    # Plot Accuracy on secondary y-axis
                    source_acc = [
                        entry.source_accuracies.get(source, np.nan) for entry in self.entries
                    ]
                    ax2_source = ax1_source.twinx()
                    ax2_source.plot(epochs, source_acc, label=f"Accuracy @{k}", color="green")
                    ax2_source.set_ylabel(f"Accuracy @{k}")
                    ax2_source.set_ylim(0, 1.0 * headroom_multiplier)  # Assuming accuracy ranges between 0 and 1
                    ax2_source.legend(loc="upper right")
                
                # Update the title
                source_title = f"{title} - {label_to_source(source)}: Train vs Validation Loss"
                if self.has_accuracy:
                    source_title += f" with Accuracy @{k}"
                fig_source.suptitle(source_title)
                figures.append(fig_source)
        
        # Adjust layout to accommodate suptitles
        for f in figures:
            f.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
        
        return figures

    def dump_graph(self, title, outpath):
        figures = self._get_figures(title)
        _save_figures_as_collage_matplotlib(figures, outpath)

    def dump_log(self, outpath):
        with open(outpath, 'w') as f:
            json.dump([entry.to_dict() for entry in self.entries], f)
    
class GTZANLogEntry:
    def __init__(self, epoch, train_loss=None, val_loss=None,
                 accuracy=None, source_accuracies=None,
                 source_train_losses=None, source_val_losses=None, top_k=None):
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.accuracy = accuracy
        self.top_k = top_k

        self.source_accuracies = {} if source_accuracies is None else source_accuracies
        self.source_train_losses = {} if source_train_losses is None else source_train_losses
        self.source_val_losses = {} if source_val_losses is None else source_val_losses

    def to_dict(self):
        base_dict = {
            'epoch': self.epoch,
        }

        # Add optional fields
        if self.train_loss is not None:
            base_dict['train_loss'] = self.train_loss
        if self.val_loss is not None:
            base_dict['val_loss'] = self.val_loss
        if self.accuracy is not None:
            # Convert numpy arrays to lists
            if isinstance(self.accuracy, np.ndarray):
                base_dict['accuracy'] = self.accuracy.tolist()
            else:
                base_dict['accuracy'] = self.accuracy
        if self.top_k is not None:
            base_dict['top_k'] = self.top_k
        if len(self.source_accuracies) > 0:
            # Convert numpy arrays in source_accuracies to lists
            base_dict['source_accuracies'] = {
                source: acc.tolist() if isinstance(acc, np.ndarray) else acc
                for source, acc in self.source_accuracies.items()
            }
        if len(self.source_train_losses) > 0:
            base_dict['source_train_losses'] = self.source_train_losses
        if len(self.source_val_losses) > 0:
            base_dict['source_val_losses'] = self.source_val_losses
        
        return base_dict


    def print_train_loss(self):
        #Print total loss
        print(f"Epoch {self.epoch + 1} - Training loss: {self.train_loss:.3f}")

        #Print source losses on one line
        if len(self.source_train_losses) > 0:
            line = "By source:"
            for source, loss in self.source_train_losses.items():
                line += f" {label_to_source(source)}: {loss:.3f},"
            line = line[:-1] #Remove last comma
            print(line)

    def print_val_loss(self):
        #Print total loss
        print(f"Epoch {self.epoch + 1} - Validation loss: {self.val_loss:.3f}")

        #Print source losses on one line
        if len(self.source_val_losses) > 0:
            line = "By source:"
            for source, loss in self.source_val_losses.items():
                line += f" {label_to_source(source)}: {loss:.3f},"
            line = line[:-1]
            print(line)

    def print_accuracy(self):
        if self.accuracy is None or self.top_k is None:
            return
        
        if isinstance(self.top_k, int):
            print(f"Epoch {self.epoch + 1} - Validation accuracy: {self.accuracy:.3f}")

            if len(self.source_accuracies) > 0:
                line = "By source:"
                for source, acc in self.source_accuracies.items():
                    line += f" {label_to_source(source)}: {acc}:.3f,"
                line = line[:-1]
                print(line)
        else:
            # Overall
            msg = f"Epoch {self.epoch + 1} - Validation accuracy:"
            for i, k in enumerate(self.top_k):
                msg += f" @{k}: {self.accuracy[i]:.3f},"
            msg = msg[:-1]
            print(msg)

            # By source
            if len(self.source_accuracies) > 0:
                for source, acc in self.source_accuracies.items():
                    line = f"For source {source}:"
                    for i, k in enumerate(self.top_k):
                        line += f" @{k}: {acc[i]:.3f},"
                    line = line[:-1]
                    print(line)

    
