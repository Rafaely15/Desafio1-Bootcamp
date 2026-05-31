# Metodologia da versao web operacional

Esta versao foi pensada para o uso real no processo de picking. O colaborador nao precisa conhecer os parametros do modelo. Ele apenas se identifica, segue orientacoes de captura, tira a foto e confirma a contagem.

Cada registro e associado ao funcionario, pedido ou lote e data, permitindo rastreabilidade e auditoria do processo. As correcoes manuais tambem sao salvas, criando uma base historica que pode ser usada futuramente para treinar uma solucao supervisionada com YOLO.

## Decisoes de produto

- A interface normal esconde parametros tecnicos.
- O fluxo principal prioriza uso em smartphone.
- A foto pode vir da camera do navegador ou da galeria.
- A confirmacao humana continua no processo para reduzir risco operacional.
- A area tecnica fica separada e protegida por senha simples para o prototipo.

## Dados salvos

O sistema salva cada conferencia em SQLite e atualiza um CSV diario. Tambem salva a imagem original e a imagem anotada com a deteccao.

Estrutura:

```text
database/contador_parafusos.db
records/YYYY-MM-DD/originals/
records/YYYY-MM-DD/detections/
records/YYYY-MM-DD/exports/
```

## Papel do OpenCV

O pipeline OpenCV e uma baseline explicavel: permite entender mascaras, componentes, peaks e erros de segmentacao. Isso e adequado para demonstracao e para dataset pequeno.

## Evolucao com YOLO

Quando houver volume suficiente de imagens confirmadas e corrigidas, o proximo passo recomendado e anotar os parafusos e treinar um detector YOLO. O YOLO deve melhorar a robustez em fundos variados, iluminacao irregular, objetos parcialmente sobrepostos e mudancas de escala.

