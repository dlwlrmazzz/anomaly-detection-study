### 특징
- 각 time-step마다 prior를 다르게 했다.
- dynamic threshold를 통해서 anomaly score를 anomaly인지 아닌지 판단했다.  
  - latent Z를 input으로 하고 그때 나온 anomaly score를 output으로 해서 SVR fitting
  - 이를 통해 inference할 때, latent값을 넣어서 threshold값을 구한 뒤에 anomaly score가 그 값보다 큰 경우 anomaly라고 판단