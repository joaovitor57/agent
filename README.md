O conteúdo abrange Processamento de Linguagem Natural (NLP), Redes Neurais Convolucionais (CNN), Redes Recorrentes (RNN/LSTM) e a Arquitetura Transformer.

---

### Parte 1: Resumo dos Conceitos Principais

#### 1. Fundamentos de NLP (Natural Language Processing)
 O NLP visa permitir que computadores compreendam e gerem linguagem humana.
* **Pré-processamento:**
    *  **Tokenização:** Divisão do texto em unidades menores (tokens), que podem ser palavras ou sub-palavras.
    *  **Stopwords:** Palavras irrelevantes (ex: "de", "para") removidas para reduzir ruído e dimensionalidade.
    * **Normalização:**
        *  *Stemming:* Remove sufixos/prefixos para chegar à raiz (pode gerar palavras inválidas, ex: "cantaram" -> "cant").
        *  *Lemmatization:* Reduz a palavra à sua forma canônica (lema), resultando em uma palavra válida.
* **Representação Vetorial (Embeddings):**
    *  Textos precisam ser convertidos em números.
    *  **Word2Vec (CBOW):** Uma rede neural simples que usa o contexto (palavras vizinhas) para prever uma palavra alvo.  A entrada é *One-hot encoding* e os pesos da camada oculta tornam-se o vetor da palavra (embedding).

#### 2. Redes Neurais Convolucionais (CNN)
 Focadas em reconhecimento de padrões e imagens. A arquitetura  possui três pilares:
*  **Camada Convolucional:** Funciona como extrator de características (features).  Usa filtros (kernels) que percorrem a imagem realizando somas de produtos para identificar padrões (ex: bordas, formas).
*  **Camada de Pooling:** Reduz a dimensionalidade e seleciona as características mais fortes (ex: Max Pooling pega o maior valor).
*  **Fully Connected (FC):** Uma rede neural densa (MLP) no final que classifica a imagem baseada nas features extraídas.

#### 3. Redes Neurais Recorrentes (RNN) e LSTM
 Utilizadas para dados sequenciais onde o histórico importa (ex: texto, séries temporais).
*  **RNN (Recurrent Neural Network):** Possui um loop onde a saída atual depende do input atual e do estado oculto anterior ($h_{t-1}$).
    * *Problema:* **Vanishing Gradient** (Desaparecimento do Gradiente).  Em sequências longas, a multiplicação sucessiva de pesos $<1$ faz a rede "esquecer" o início da frase.
* **LSTM (Long Short-Term Memory):** Resolve o problema da memória curta da RNN .  Possui uma célula de memória ($C_t$) e portas (gates) que controlam o fluxo de informação:
    *  *Forget Gate:* Decide o que esquecer (usa sigmoide).
    *  *Input Gate:* Decide o que adicionar à memória (usa sigmoide e tangente hiperbólica).
    *  *Output Gate:* Decide a saída baseada na memória atualizada.

#### 4. Arquitetura Transformer
Baseada no mecanismo de atenção, dispensando a recorrência sequencial. Divide-se em Encoder e Decoder.
* **Conceitos Chave:**
    *  **Positional Encoding:** Como não há recorrência, soma-se um vetor de posição ao embedding para que a rede saiba a ordem das palavras.
    *  **Self-Attention:** Permite que um token "olhe" para todos os outros da frase para entender o contexto.  Utiliza três vetores para cada token: **Q** (Query/Busca), **K** (Key/Chave) e **V** (Value/Conteúdo).
        *  O score de atenção é calculado por $score_{ij} = \frac{Q \cdot K^T}{\sqrt{d_k}}$, passado por uma *softmax* e multiplicado por $V$.
*  **Encoder:** Gera uma representação contextual do texto.  Usa *Self-Attention* bidirecional (olha todo a frase).
*  **Decoder:** Gera o texto de saída (ex: tradução).
    *  *Masked Self-Attention:* O decoder só pode olhar para tokens anteriores (o futuro é mascarado com $-\infty$).
    *  *Cross-Attention:* O Decoder usa suas próprias Queries (Q), mas busca as Keys (K) e Values (V) gerados pelo Encoder.

---

### Parte 2: Revisão de Estudos para a Prova

Aqui está um roteiro de perguntas e tópicos que você deve dominar, baseado nos objetivos de aula dos slides.

#### Tópico 1: Processamento de Texto (NLP)
1.  **Explique a diferença entre Stemming e Lemmatization.**
    *  *Dica:* Foque no resultado final (raiz vs. palavra válida).
2.  **Por que removemos Stopwords?**
    *  *Dica:* Pense em "ruído" e dimensionalidade.
3.  **Como funciona o algoritmo CBOW (Word2Vec)?**
    *  *Dica:* Lembre-se que ele usa o contexto para prever a palavra central e como os pesos da camada oculta se tornam o embedding.

#### Tópico 2: Visão Computacional (CNN)
4.  **Qual a função dos filtros na camada de convolução?**
    *  *Dica:* Extração de características (features) através de operações matriciais.
5.  **O que a camada de Pooling (ex: Max Pooling) faz com a dimensionalidade da imagem?**
    *  *Dica:* Redução e seleção das características mais fortes.

#### Tópico 3: Sequências e Memória (RNN/LSTM)
6.  **Qual a limitação principal das RNNs simples ao processar frases muito longas?**
    *  *Dica:* Problema do Vanishing Gradient (memória curta).
7.  **Na arquitetura LSTM, qual o papel do "Forget Gate"?**
    *  *Dica:* Decidir a porcentagem da memória longa antiga que será mantida.
8.  **Diferencie $h_t$ (memória curta) de $C_t$ (memória longa) na LSTM.**
    *  *Dica:* $C_t$ percorre toda a cadeia com regulações lineares, enquanto $h_t$ é a saída processada para o momento atual.

#### Tópico 4: Transformers (Encoder/Decoder)
9.  **Por que o Positional Encoding é necessário em Transformers e não em RNNs?**
    *  *Dica:* Transformers processam tokens em paralelo, não em loop, então não têm noção intrínseca de ordem.
10. **Explique a analogia de Q (Query), K (Key) e V (Value) no Self-Attention.**
    *  *Dica:* Q é o que estou buscando, K é o que o outro token oferece para busca, V é o conteúdo que será entregue se houver "match".
11. **Qual a diferença crucial entre o Self-Attention do Encoder e do Decoder?**
    *  *Dica:* O Decoder não pode ver o futuro, por isso usa mascaramento (masked self-attention).
12. **O que é Cross-Attention?**
    *  *Dica:* O momento em que o Decoder (Q) consulta o que foi processado pelo Encoder (K, V).

---