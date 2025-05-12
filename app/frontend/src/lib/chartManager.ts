import { Chart } from 'chart.js/auto';

export interface TrainingData {
  current_epoch: number,
  total_epochs: number,
  train_loss: number[],
  val_loss: number[],
  ETA: number,
  is_model_trained: boolean
}

export class ChartManager {
  chart: Chart
  uiManager: UIManager

  constructor(canvas: HTMLCanvasElement, uiManager: UIManager) {
    this.uiManager = uiManager

    this.chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          this.createDataset("Training Loss", "rgb(17, 182, 251)") as never,
          this.createDataset("Validation Loss", "rgb(167, 103, 245)") as never
        ]
      },
      options: {
        backgroundColor: 'rgba(0, 0, 0, 0)',
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    })
  }

  private createDataset(label: string, color: string) {
    return {
      type: 'line',
      label: label,
      data: [],
      borderColor: color,
      tension: 0.1
    }
  }

  update(data: TrainingData) {
    const percentage = data.current_epoch / data.total_epochs * 100;
    this.uiManager.updateProgress(percentage || 0);

    if (this.chart.data.labels?.length === 0) {
      this.chart.data.labels = Array.from({ length: data.current_epoch }, (_, i) => i + 1) as never[];
      this.chart.data.datasets[0].data = data.train_loss;
      this.chart.data.datasets[1].data = data.val_loss;
      this.chart.update();
      this.uiManager.updateETA(data.ETA);
    } else {
      const lastEpoch = this.getLastEpoch();

      if (lastEpoch < data.current_epoch) {
        this.chart.data.labels =  Array.from({ length: data.current_epoch }, (_, i) => i + 1)
        this.chart.data.datasets[0].data = data.train_loss
        this.chart.data.datasets[1].data = data.val_loss
        this.chart.update();
        this.uiManager.updateETA(data.ETA);
      }
    }
  }

  private getLastEpoch(): number {
    return Math.max(...(this.chart.data.labels as number[] || [0]))
  }
}

export class UIManager {
  private ETATimer: any = null;
  private ETA = 0;
  public modelTrainingProgress: HTMLProgressElement;
  public modelTrainingProgressText: HTMLSpanElement;
  public modelTrainingETA: HTMLSpanElement;

  constructor(modelTrainingProgress: HTMLProgressElement, modelTrainingProgressText: HTMLSpanElement, modelTrainingETA: HTMLSpanElement) {
    this.modelTrainingProgress = modelTrainingProgress
    this.modelTrainingProgressText = modelTrainingProgressText
    this.modelTrainingETA = modelTrainingETA
  }

  updateProgress(percentage: number) {
    this.modelTrainingProgress.value = percentage
    this.modelTrainingProgressText.innerText = Math.round(percentage) + "%";
  }

  updateETA(ETA: number) {
    this.ETA = ETA
    if (!this.ETATimer) {
      this.ETATimer = setInterval(() => {
        this.ETA = Math.max(0, this.ETA - 1)
        this.displayETA()
      }, 1000)
    }
    this.displayETA()
  }
  
  displayETA() {
    const etaMinutes = Math.floor(this.ETA / 60);
    const etaSecondsLeft = Math.floor((this.ETA % 60));

    this.modelTrainingETA.innerText = `ETA: ${etaMinutes}m ${etaSecondsLeft}s`;
  }
}