# Calculadora Estatística — Queda da Régua

Script Python para analisar as medidas de queda de uma régua coletadas por
quatro participantes. O programa calcula estatísticas descritivas e
dispersão, produz tabelas de frequência, estima o tempo de reação e a
distância percorrida por um veículo a 100 km/h e gera gráficos locais.

## Requisitos

- Python 3.8 ou superior.
- Nenhuma biblioteca de terceiros: o programa usa apenas a biblioteca padrão
  do Python.

## Executar

Abra um terminal nesta pasta e execute:

```powershell
python Calculadora.py
```

Em alguns ambientes Windows, pode ser necessário usar `py Calculadora.py` ou
informar o caminho do interpretador Python instalado.

## Dados de entrada

As 40 medidas em centímetros de cada participante estão na variável
`dados_experimento`, no início de `Calculadora.py`:

- Lucas C
- Lucas K
- Felipe
- Mauricio

Para corrigir ou trocar uma medida, edite o valor na posição correspondente da
lista. A primeira posição representa a medida 1 e a última, a medida 40. Se
uma observação estiver ausente, use `None` naquela posição, sem remover o
elemento: assim o gráfico de linhas preserva a ordem e o número original da
medida. O valor zero é aceito como observação válida.

O programa valida que cada participante tenha exatamente 40 posições e que
os valores informados sejam numéricos, finitos e não negativos. A lista
original de Mauricio continha um elemento vazio após o valor 14; esse vazio
foi tratado como erro de digitação, conforme confirmação do usuário, e os 40
valores numéricos foram preservados.

## Resultados estatísticos

Para cada participante, o programa exibe:

- número de medidas preenchidas, média, mediana e moda(s);
- desvio padrão amostral, calculado com divisor `n - 1`;
- quartis Q1, Q2 e Q3, com interpolação linear; Q2 corresponde à mediana;
- amplitude interquartil `AIQ = Q3 - Q1`;
- limites de outliers `LI = Q1 - 1,5 × AIQ` e
  `LS = Q3 + 1,5 × AIQ`, incluindo o número da medida discrepante;
- coeficiente de assimetria e excesso de curtose de Fisher, com correção
  amostral;
- tempo de queda médio e distância estimada do veículo.

Quando as observações forem insuficientes ou não tiverem variação, o
coeficiente que não puder ser calculado será mostrado como `NaN`. Se todos os
valores forem únicos, o programa informa “Sem moda”.

### Tabelas de frequência

As medidas são agrupadas em seis classes de mesma largura. O limite inferior
de cada classe é inclusivo; o superior é exclusivo, exceto na última classe,
que inclui o valor máximo. São mostradas frequência absoluta, frequência
relativa e suas versões acumuladas.

## Aplicações físicas

As distâncias coletadas em centímetros são convertidas para metros. O tempo
de queda de cada observação é calculado por:

```text
t = sqrt(2 × d / g)
g = 9,80665 m/s²
```

O modelo supõe que a régua parte do repouso e despreza a resistência do ar.
O tempo apresentado é a média dos tempos calculados individualmente. A
velocidade de 100 km/h é convertida para m/s (`100 / 3,6`) e multiplicada
pelo tempo médio para estimar a distância percorrida pelo veículo durante
esse intervalo.

## Gráficos gerados

Ao executar o programa, os arquivos abaixo são criados na mesma pasta de
`Calculadora.py`:

- `grafico_sequencia_temporal.svg`: medidas 1 a 40 de todos os participantes;
- `histograma_lucas_c.svg`, `histograma_lucas_k.svg`,
  `histograma_felipe.svg` e `histograma_mauricio.svg`: histogramas
  individuais;
- `boxplot_lucas_c.svg`, `boxplot_lucas_k.svg`, `boxplot_felipe.svg` e
  `boxplot_mauricio.svg`: boxplots individuais;
- `grafico_boxplots_comparativo.svg`: comparação dos boxplots;
- `graficos_calculadora.html`: índice visual que reúne os gráficos.

Abra `graficos_calculadora.html` em um navegador para visualizar todos os
gráficos juntos. Cada SVG também pode ser aberto separadamente em um navegador
ou aplicativo compatível. Os arquivos são recriados quando o script é
executado novamente.

## Arquivos do projeto

- `Calculadora.py`: dados, validações, cálculos, tabelas e geração de gráficos.
- `README.md`: instruções de uso, requisitos e convenções da análise.
