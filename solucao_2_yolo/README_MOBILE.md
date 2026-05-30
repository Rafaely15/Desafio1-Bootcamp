# Exportação Mobile

O caminho recomendado inicial é smartphone + servidor: o funcionário acessa a página web, envia a foto, e o servidor executa YOLO11. Esse modo centraliza dados, facilita manutenção, permite usar modelo mais preciso e evita depender da potência do celular.

## Treino no PC, Inferência no Celular

Não é recomendado treinar no smartphone. O treinamento exige GPU, muita memória, tempo e gerenciamento de dataset. O celular deve apenas rodar inferência quando houver necessidade de uso offline.

Modelos menores, como `yolo11n.pt`, costumam ser melhores para celular. Modelos maiores podem ser melhores no servidor.

## Exportar para Android com TFLite

```bash
python scripts/export_tflite.py
```

Comando equivalente:

```bash
yolo export model=runs_screws/yolo11_screws/weights/best.pt format=tflite
```

## Exportar para NCNN

```bash
python scripts/export_ncnn.py
```

Comando equivalente:

```bash
yolo export model=runs_screws/yolo11_screws/weights/best.pt format=ncnn
```

NCNN pode ser uma boa opção para Android, iOS e dispositivos embarcados.

## Exportar para iPhone/iPad com CoreML

```bash
python scripts/export_coreml.py
```

Comando equivalente:

```bash
yolo export model=runs_screws/yolo11_screws/weights/best.pt format=coreml
```

## Qual Formato Usar

- Android: TFLite ou NCNN.
- iPhone/iPad: CoreML.
- Servidor: `best.pt` diretamente.

## Limitações

- celular fraco pode ser lento;
- modelos grandes consomem mais memória;
- objetos pequenos exigem boa resolução;
- iluminação afeta muito o resultado;
- parafusos pequenos podem exigir imagens maiores e boa iluminação;
- reflexos metálicos e sobreposição reduzem a qualidade da contagem.

## Recomendação

Comece com aplicação web + servidor. Depois, se realmente precisar trabalhar sem internet/rede local, crie uma versão offline mobile usando o modelo exportado.
