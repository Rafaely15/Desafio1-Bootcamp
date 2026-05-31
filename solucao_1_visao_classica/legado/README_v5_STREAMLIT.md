# Projeto Parafusos V5 — Streamlit + Segmentação Adaptativa

Esta versão implementa a melhoria sugerida para o Desafio 1:

1. **Correção do caso img2** com segmentação adaptativa ao ruído de fundo.
2. **Interface Web Streamlit** para upload de novas imagens e teste das imagens do dataset.

## O que mudou na segmentação

A função `segment_adaptive()` foi adicionada ao `screw_counter.py`.

Ela calcula, antes da decisão final, as métricas:

- `noise_ratio = n_tiny_comps / n_total_comps`
- `n_big`, número de componentes com área entre `min_area` e `image_area * 0.08`
- `n_tiny`, componentes pequenos
- `n_bg`, componentes grandes demais

O blur adaptativo é ativado quando:

```python
noise_ratio > 0.90 and n_big >= 3
```

Isso corrige o caso da `img2`, em que o fundo texturizado gerava muitos fragmentos falsos.

## Resultados nos exemplos incluídos

Com parâmetros padrão:

| Imagem | Contagem |
|---|---:|
| img1.jpg | 8 |
| img2.jpg | 1 |
| img3.jpg | 4 |
| img4.jpg | 2 |
| img5.jpg | 8 |
| img6.jpg | 5 |

## Como instalar

```bash
pip install -r requirements.txt
```

## Como rodar

```bash
streamlit run app.py
```

Acesse no navegador:

```text
http://localhost:8501
```

## Como abrir no smartphone

Computador e celular precisam estar na mesma rede Wi-Fi.

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

No Windows, descubra o IP do computador:

```bash
ipconfig
```

No celular, abra:

```text
http://IP_DO_COMPUTADOR:8501
```

## Interface

A interface possui:

- upload de uma ou várias imagens;
- teste direto das imagens em `data/raw/`;
- sliders para `ref_area` e `min_area`;
- opção `Usar ref auto`;
- visualização em quatro painéis:
  - Original;
  - Máscara;
  - DT + Peaks;
  - Detecção final;
- histórico da sessão;
- download do CSV com resultados.

## Observação metodológica

Esta solução continua sendo visão computacional clássica, adequada como baseline interpretável para poucas imagens. Para generalizar em cenários mais variados, a evolução recomendada é coletar e anotar imagens para treinar um detector supervisionado, como YOLO.
