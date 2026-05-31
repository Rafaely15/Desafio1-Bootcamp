# Relatorio de metricas

- Imagens avaliadas: 5
- MAE: 0.000
- Acuracia exata: 100.0%
- Vies medio: 0.000
- Supercontagens: 0
- Subcontagens: 0

## Tabela de resultados

| filename   |   y_true |   y_pred |   absolute_error |   signed_error | mode_used                  | segmentation_mode   |   ref_area |   min_area | watershed_used   | blur_used   |   noise_ratio |   n_components |   n_big_components |   n_tiny_components |   processing_time_ms |
|:-----------|---------:|---------:|-----------------:|---------------:|:---------------------------|:--------------------|-----------:|-----------:|:-----------------|:------------|--------------:|---------------:|-------------------:|--------------------:|---------------------:|
| img1.jpg   |        8 |        8 |                0 |              0 | detector_contornos         | modo_multiescala    |    3333    |        300 | False            | True        |        0.8182 |             44 |                  8 |                  36 |              2439.82 |
| img2.jpg   |        1 |        1 |                0 |              0 | detector_contornos         | modo_cinza          |    4021    |        300 | False            | True        |        0.9429 |           1104 |                 62 |                1041 |              2047.97 |
| img3.jpg   |        4 |        4 |                0 |              0 | detector_watershed         | modo_multiescala    |    4223    |        300 | False            | True        |        0.8889 |             36 |                  4 |                  32 |              1012.86 |
| img4.jpg   |        2 |        2 |                0 |              0 | detector_fragmentos_dbscan | modo_multiescala    |    4377    |        300 | False            | True        |        0.8462 |             13 |                  2 |                  11 |               708.22 |
| img5.jpg   |       10 |       10 |                0 |              0 | detector_contornos         | modo_multiescala    |    4021    |        300 | False            | True        |        0.9394 |             33 |                  2 |                  31 |              1682.34 |
| img6.jpg   |      nan |        5 |              nan |            nan | detector_contornos         | modo_multiescala    |    1809.45 |        300 | False            | True        |        0      |              5 |                  5 |                   0 |               101.45 |

## Imagens com maior erro

| filename   |   y_true |   y_pred |   absolute_error |   signed_error | mode_used                  | segmentation_mode   |   ref_area |   min_area | watershed_used   | blur_used   |   noise_ratio |   n_components |   n_big_components |   n_tiny_components |   processing_time_ms |
|:-----------|---------:|---------:|-----------------:|---------------:|:---------------------------|:--------------------|-----------:|-----------:|:-----------------|:------------|--------------:|---------------:|-------------------:|--------------------:|---------------------:|
| img1.jpg   |        8 |        8 |                0 |              0 | detector_contornos         | modo_multiescala    |       3333 |        300 | False            | True        |        0.8182 |             44 |                  8 |                  36 |              2439.82 |
| img2.jpg   |        1 |        1 |                0 |              0 | detector_contornos         | modo_cinza          |       4021 |        300 | False            | True        |        0.9429 |           1104 |                 62 |                1041 |              2047.97 |
| img3.jpg   |        4 |        4 |                0 |              0 | detector_watershed         | modo_multiescala    |       4223 |        300 | False            | True        |        0.8889 |             36 |                  4 |                  32 |              1012.86 |
| img4.jpg   |        2 |        2 |                0 |              0 | detector_fragmentos_dbscan | modo_multiescala    |       4377 |        300 | False            | True        |        0.8462 |             13 |                  2 |                  11 |               708.22 |
| img5.jpg   |       10 |       10 |                0 |              0 | detector_contornos         | modo_multiescala    |       4021 |        300 | False            | True        |        0.9394 |             33 |                  2 |                  31 |              1682.34 |

## Analise de subcontagem

Nao houve subcontagem nos labels disponiveis.

## Analise de supercontagem

Nao houve supercontagem nos labels disponiveis.

## Limitacoes do OpenCV

A abordagem classica e leve, explicavel e util como baseline, mas depende de contraste, escala e textura. Em aglomerados densos, objetos encostados podem virar um unico blob; em fundos texturizados, um objeto pode se fragmentar.

## Evolucao com YOLO

Com feedback manual e novas imagens anotadas, YOLO e a evolucao natural para generalizacao robusta, pois aprende aparencia, escala, orientacao e contexto diretamente dos exemplos.