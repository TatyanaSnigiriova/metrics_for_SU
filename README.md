# metrics_for_SU
calculate_metrics.py - Реализует подсчет следующих метрик качества: 
* MSE (Mean Squared Error)
* MAE (Mean Absolute Error)
* PSNR (Peak signal-to-noise ratio)
* MII (Mutual information Index)
* NCC (Normalized Cross-Correlation)
* SIIM (Structural similarity) 

Следующие 2 файла предназначены для тестирования устойчивости метрик к некоторым преобразованиям (сдвигам):  
* generate_datasets_for_testing_metrics.py - Генерирует наборы с различными значениями сдвигов для 1 входного изображения;  
* test_metrics.py - Тестирование метрик на сгенерированных наборах.  

## Настройка окружения: 
https://drive.google.com/file/d/1cARntiW1oEgR_ZU0ltBj-ekWnN0BS2SL/view?usp=sharing
* Используются TF v.1
