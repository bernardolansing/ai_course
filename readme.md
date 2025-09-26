# T1
## Better parameters
- b = -3.3523656365077175
- w = 1.1509374496448106
- alpha = 0.01
- iterations = 1000

## Complexidade dos datasets
Quatro datasets foram ajustados.
- O MNIST é um conjunto de figuras monocromáticas 28x28 de dígitos manuscritos (10 classes).
- O Fashion MNIST também é composto de imagens 28x28 monocromáticas, mas sobre peças de vestuário (10 classes).
- O CIFAR-10 é um conjunto de imagens 32x32 com 3 canais (RGB). As imagens podem ser classificadas em 10 categorias (classes) distintas.
- O CIFAR-100 é a mesma coisa que o CIFAR-10, porém sendo de 100 classes.

Os datasets foram listados na ordem crescente de complexidade. O Fashion MNIST é mais complexo que seu sucessor porque as peças de vestuário formam imagens bem mais ricas em detalhes, havendo mais semelhança entre instâncias de classes distintas.

O desafio aumenta consideravelmente no CIFAR-10, porque agora temos uma resolução um pouco maior e três canais de cor. Isto já mais do que triplica o tamanho dos dados. Pode, aliás, mais do que triplicar a complexidade do treino, devido às correlações internas que surgem quando a cor é levada em consideração. Por exemplo, se quisermos que o algoritmo detecte sapos, provavelmente ele deverá dar uma importância grande à cor verde estar presente na imagem. Mas, para a cor verde estar presente, o canal cromático G deve estar num valor alto, ao mesmo tempo em que o canal B está num valor médio e o canal R num valor baixo. Ou seja, há uma forte codependência interna nos dados, e detectar isto demanda mais filtros e mais camadas.

Por fim, há o CIFAR-100, que é a mesma coisa que o CIFAR-10, mas com mais classes. Isso, naturalmente, faz dele o treinamento mais laborioso dentre os quatro.

## Performance dos modelos
- CIFAR-10: **70,67%**
- CIFAR-100: **31,80%**

Para os conjuntos CIFAR, percebemos que não adiantava colocar muito filtros. O treinamento se tornava demorado demais e a acurácia começava a diminuir. Colocar muitas camadas também não ajudou. Começamos a encontrar uma melhoria sensível de acurácia (em torno dos 50% a 70%) quando colocávamos duas camadas convolucionais com menos de 100 filtros cada e duas ou três camadas de perceptrons, igualmente com menos de 100 neurônios, e uma camada de max pooling 2x2 ligando as duas etapas. Com isso, nossos treinamentos estavam alcançando cerca de 60% de acurácia. No CIFAR-100, no entanto, começamos a enfrentar problemas sérios de overfitting. A acurácia de teste era menos da metade da acurácia de treinamento.

Coisas que não funcionaram:
- Trocar a função da última camada convolucional para sigmoide causou perdas desastrosas de acurácia. Ficou em torno de 1%.
- Aumentar a learning rate não trouxe muitas mudanças, só houve uma ligeira diminuição nas acurácias.

Ao adicionar uma camada de max pooling 2x2 entre as camadas convolucionais, o problema do overfitting foi completamente sanado. Talvez porque ele ajuda a diminuir o acoplamento entre elas, tornando o modelo menos "detalhista". A acurácia final do modelo não melhorou, no entanto.

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/4e48fae6-0c1c-411a-953a-273ae44bce37" />

Provavelmente, para obter mais acurácia, deveríamos recorrer a técnicas um pouco mais avançadas, como normalização ou _data augmentation_. Além disso, notou-se que as camadas de _pooling_ influenciavam sensivelmente o desempenho do modelo, especialmente no quesito _overfitting_. Possivelmente, empregar outras técnicas de _pooling_ (que não o "max") também traria melhores resultados.

- MNIST: **97,59%**
- Fashion MNIST: **90,73%**

As instâncias dos MNIST são "arrays" de 784 valores booleanos. Isso o torna tão simples que não foi necessário adicionar camadas convolucionais. O refinamento do modelo foi executado adicionando-se camadas de perceptrons até que a precisão começasse a cair, ao mesmo tempo em que controlava-se o número de neurônios em cada uma delas de modo a afunilar-na.

Quanto ao Fashion MNIST, devido ao aumento de complexidade, consideramos que valia a pena adicionar uma camada convolucional. Após um breve _tweaking_ nas configurações dela e na camada de _pooling_ que a sucede, conseguimos melhorar o resultados a um nível satisfatório. É válido observar que o ganho de acurácia estagnou após a quinta _epoch_; a partir deste ponto o modelo apenas começou a sobreajustar-se aos dados.

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/6b39dcc5-7437-487b-8ea9-e4cff8dbbfbf" />
