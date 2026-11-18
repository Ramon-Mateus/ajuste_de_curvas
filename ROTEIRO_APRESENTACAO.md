# Roteiro da apresentação (Projeto_U2_simples.ipynb)

Como usar: abra o notebook e role até cada parte. O texto **entre aspas** é o que você fala. O que está em *[colchetes]* é o que você faz ou aponta na tela.

Tempo total: uns 8 a 10 minutos.

---

## Abertura (célula de título)

*[Mostra o título no topo do notebook.]*

> "O objetivo do projeto é achar um modelo que imite um sistema. A gente tem a entrada `u` e duas saídas medidas, `x` e `y`, e as saídas têm ruído.
>
> O enunciado diz o formato do modelo: o valor da saída agora depende dos dois valores anteriores dela e da entrada agora e um instante antes.
>
> Eu encontrei essa função de dois jeitos: com RBF, que é uma soma de gaussianas, e com MMQ, que é uma conta linear com 4 parâmetros. Depois comparei os dois."

---

## 1. Leitura dos dados

### Célula 1 (leitura do CSV)

> "Primeiro eu leio o arquivo. Os números usam vírgula decimal, então eu troco a vírgula por ponto e transformo em número.
>
> Depois separo as colunas: `t` é o tempo, `u` é a entrada, `x` e `y` são as saídas. São 1501 amostras, uma a cada 0,02 segundo, de 0 a 30 segundos."

*[Aponta a saída: `1501 amostras`.]*

### Célula 2 (gráfico dos dados)

> "Aqui estão os dados. Em cima está a entrada `u`, que só vale +1 ou −1 e troca a cada 1 segundo.
>
> Embaixo está o `y`: quando `u` é +1, ele sobe, e quando é −1, ele desce. Ele não pula na hora, ele vai subindo e descendo aos poucos.
>
> O `x` também reage, mas tem muito mais ruído, então é mais difícil de ver o padrão."

*[Aponta: `u` em cima, `x` tremido no meio, `y` em zigue-zague embaixo.]*

---

## 2. Funções

### Célula 3

> "Aqui eu escrevi as ferramentas que vou usar o tempo todo.
>
> - `emq` é o erro médio quadrático: pego a diferença entre o medido e o previsto, elevo ao quadrado e tiro a média. Quanto menor, melhor.
> - `pearson` é a correlação: mostra se o medido e o previsto sobem e descem juntos. Vai de −1 a 1, e perto de 1 é bom.
> - `regressao_linear` é a reta da aula, resolvendo o sistema 2 por 2.
> - `minimos_quadrados` resolve a equação normal, A transposta vezes A, vezes w, igual a A transposta vezes o alvo. Ela me devolve os pesos que deixam o erro ao quadrado o menor possível. **Essa função é usada pelos dois modelos.**"

---

## 3. Montando as entradas

### Célula 4

> "Aqui eu monto a tabela que o modelo usa para aprender. Para cada instante `n`, a linha tem 4 valores: a saída um instante antes, a saída dois instantes antes, a entrada agora e a entrada um instante antes. O alvo é a saída agora.
>
> Começo em `n = 2`, porque preciso de dois valores anteriores. Por isso ficam 1499 linhas e 4 colunas."

*[Aponta a saída: `(1499, 4)`.]*

---

## 4. RBF

### Célula 5 (funções da RBF)

> "A função `gaussianas` transforma cada linha da tabela numa linha de 'quanto cada sino acende'.
>
> Para cada centro, eu calculo a distância ao quadrado entre a linha e o centro e aplico a gaussiana, exp(−d² / 2σ²). Se a linha está perto do centro, dá perto de 1. Se está longe, dá perto de 0. O sigma é a largura do sino.
>
> A primeira coluna é só de 1, para o peso w0, que é um valor fixo somado.
>
> Para escolher os centros, eu separo as linhas em 4 grupos, conforme `u[n]` e `u[n−1]`: ligado/ligado, desligado/desligado e os dois casos de troca. Separo assim porque o `u` só vale +1 ou −1, então um sino de um grupo fica longe demais das linhas do outro. Dentro de cada grupo, ordeno as linhas e pego 4 bem espaçadas. Isso dá 16 centros, e eles caem em cima dos dados."

### Célula 6 (teste do sigma)

> "Aqui testei alguns valores de sigma e olhei o erro.
>
> Com 0,2, o `y` erra bastante (0,0156). Em 0,3 ele já cai para 0,0079, e a partir de 0,5 quase não muda mais. Por isso escolhi 0,5."

*[Aponta as linhas de `y`: 0,0156 → 0,0079 → 0,0076 → 0,0074.]*

### Célula 7 (ajuste final da RBF)

> "Com sigma 0,5, eu monto a matriz dos sinos e passo para a mesma função de mínimos quadrados. Ela me devolve 17 pesos: o w0 mais um para cada um dos 16 sinos. **Esses são os parâmetros livres da RBF.**
>
> A previsão é a matriz dos sinos vezes os pesos."

*[Aponta: `centros: 16 | pesos: 17`.]*

### Célula 8 (gráfico RBF e EMQ)

> "Os pontos azuis são os dados medidos e a linha laranja é a RBF. No `y`, a curva acompanha muito bem. No `x`, ela acompanha a tendência, mas o ruído é grande.
>
> O EMQ do `x` deu 0,0122 e o do `y` deu 0,0076."

### Célula 9 (regressão linear e Pearson da RBF)

> "Aqui faço a regressão linear entre o valor ajustado e o medido. Se o modelo é bom, os pontos ficam em volta da diagonal, e a reta fica com inclinação 1 e intercepto 0.
>
> Deu exatamente isso: alfa 0 e beta 1 nas duas saídas. Isso quer dizer que o modelo não erra sempre para o mesmo lado.
>
> O Pearson deu 0,95 no `y`, que é muito bom, e 0,64 no `x`, que é mais fraco por causa do ruído."

*[Aponta a nuvem do `y`, fininha, e a do `x`, espalhada.]*

---

## 5. MMQ

### Célula 10 (parâmetros do MMQ)

> "Agora o MMQ. A diferença é que não tem sino nenhum. Eu passo a própria tabela de 4 colunas para a mesma função de mínimos quadrados e recebo 4 parâmetros: a1, a2, b0 e b1. **Esses são os parâmetros livres do MMQ.**
>
> No `y`, a1 e a2 deram uns 0,49 e 0,46. Isso mostra que o valor de agora depende bastante dos dois valores anteriores."

*[Aponta a linha `y: a1, a2, b0, b1`.]*

### Célula 11 (gráfico MMQ e EMQ)

> "O gráfico fica quase igual ao da RBF. O EMQ deu 0,0123 no `x` e 0,0075 no `y`, praticamente os mesmos valores."

### Célula 12 (regressão e Pearson do MMQ)

> "A regressão também deu inclinação perto de 1 e intercepto perto de 0. O Pearson foi 0,64 no `x` e 0,95 no `y`, igual à RBF."

---

## 6. Comparação

### Célula 13

> "Aqui está tudo junto. Os dois modelos empataram: a diferença está só na terceira ou quarta casa decimal."

*[Aponta a tabela.]*

---

## 7. Conclusões

> "Conclusões:
>
> 1. A RBF e o MMQ deram praticamente o mesmo erro. Isso mostra que o sistema é quase linear. Como o MMQ tem só 4 parâmetros e a RBF tem 17, o MMQ é a melhor escolha, porque é mais simples.
> 2. O `y` foi bem modelado, com Pearson de 0,95. O `x` foi bem pior, com 0,64, porque tem muito mais ruído. Nenhum modelo consegue prever o ruído, porque ele é aleatório.
> 3. A regressão medido × ajustado deu inclinação 1 e intercepto 0, então os modelos não têm viés.
> 4. O EMQ e o Pearson medem coisas diferentes: o EMQ mostra o tamanho do erro, e o Pearson mostra se a forma da curva foi acompanhada. Por isso usei os dois."

---

## Perguntas que o professor pode fazer

**"Por que a RBF usa a mesma equação normal do MMQ?"**
> "Porque, depois que eu escolho os centros e o sigma, cada sino vira só uma coluna de números. A previsão continua sendo matriz vezes pesos. É não linear na entrada, mas linear nos pesos."

**"Por que σ = 0,5 e não 1,0, que deu um erro um pouco menor?"**
> "A diferença é muito pequena, de 0,0076 para 0,0074. E com sigma grande os sinos ficam muito largos e parecidos uns com os outros, então o modelo perde o sentido de ter vários sinos."

**"Por que 16 centros?"**
> "Quatro por grupo de `u`. Com isso o erro já ficou no mesmo nível do MMQ. Colocar mais sinos não faria muita diferença, só aumentaria o número de pesos."

**"O que é o ruído?"**
> "É a parte aleatória da medição. Ela faz os pontos ficarem espalhados em volta do valor verdadeiro. Por isso o erro nunca chega a zero."

**"Por que o `x` foi tão pior?"**
> "Porque o `x` tem muito mais ruído em relação ao tamanho do sinal. O modelo usa os valores anteriores de `x`, que também têm ruído, então a informação que ele recebe já vem bagunçada."

**"O que é o r?" / "O que é o r²?"**
> "O r é a correlação de Pearson. O r² é quanto da variação o modelo explica: no `y`, 0,95² dá uns 0,91, então explica 91%."

**"Se não souber responder"**
> "Essa parte eu não sei explicar com certeza, mas o que eu entendi é…" *e fale o que você entendeu.* É melhor do que inventar.
