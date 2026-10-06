"""
Calculadora estatística para o experimento de queda livre da régua.

CONTEXTO E ENTRADA
------------------
Cada integrante tem até 40 medidas de distância de queda da régua, em
centímetros, ordenadas conforme foram coletadas. A posição 1 da lista é a
medida 1 e assim por diante. O código preserva essa posição nos gráficos;
None pode representar uma medida ausente sem deslocar as seguintes.

As listas neste arquivo contêm os dados fornecidos para Lucas C, Lucas K,
Felipe e Mauricio. No caso de Mauricio, a posição vazia escrita após a medida
14 foi tratada como erro de digitação, conforme confirmação do usuário; os
40 valores numéricos foram preservados na sequência.

REQUISITOS ATENDIDOS E CONVENÇÕES
---------------------------------
* Estatísticas e frequências ignoram posições None. Valores iguais a zero são
  mantidos, pois são dados observados e não ausência de medida.
* A tabela de frequências e os histogramas usam seis classes de mesma largura.
  Nas classes impressas, o limite inferior é inclusivo e o superior exclusivo,
  exceto na última classe, que inclui também seu limite superior.
* Quartis são calculados pelo método linear padrão de numpy.percentile;
  Q2 é a mediana e AIQ = Q3 - Q1. Outliers seguem os limites Q1 - 1,5*AIQ
  e Q3 + 1,5*AIQ.
* O desvio padrão usa ddof=1 (amostral). Assimetria e curtose usam as versões
  corrigidas para amostra de scipy.stats; a curtose é o excesso de Fisher
  (curtose normal igual a zero). Se não houver amostra/variação suficiente,
  o resultado correspondente é NaN em vez de um número enganoso.
* A distância é convertida de centímetros para metros (divisão por 100).
  Para cada observação, t = sqrt(2*d/g), assumindo queda livre a partir do
  repouso e desprezando resistência do ar. O tempo exibido é a média desses
  tempos individuais. A distância do veículo é 100/3,6 m/s multiplicado
  por esse tempo médio.
* São salvos três PNGs na pasta de execução: sequência temporal, histogramas
  individuais e boxplots comparativos. A exibição também depende do backend
  gráfico instalado/configurado no ambiente.

DEPENDÊNCIAS
------------
Instale numpy, pandas, matplotlib, seaborn e scipy no mesmo Python usado para
executar este arquivo. Esta implementação fica intencionalmente em um único
arquivo, conforme solicitado.
"""

import math
import numbers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

NUM_MEDIDAS = 40
G_GRAVIDADE = 9.80665  # m/s²
VELOCIDADE_KMH = 100.0
VELOCIDADE_MS = VELOCIDADE_KMH / 3.6

# Cada lista tem 40 posições em ordem de coleta. Use None apenas para uma
# medida realmente ausente; não remova a posição, para preservar a sequência.
dados_experimento = {
    "Lucas C": [
        10, 10, 16, 14, 4, 10, 13, 12, 9, 9,
        10, 4, 14, 18, 15, 10, 13, 19, 15, 17,
        15, 9, 15, 28, 9, 9, 2, 9, 10, 16,
        12, 12, 9, 18, 15, 14, 11, 7, 17, 15,
    ],
    "Lucas K": [
        20, 18, 21, 18, 16, 19, 3, 17, 10, 23,
        17, 9, 16, 11, 16, 12, 18, 11, 22, 20,
        14, 15, 17, 17, 21, 20, 16, 15, 8, 21,
        11, 15, 19, 21, 17, 23, 15, 16, 23, 28,
    ],
    "Felipe": [
        16, 14.5, 27, 0, 13.5, 14, 12, 12, 5, 8,
        13, 10, 10, 8, 11.5, 9, 18, 10, 9, 11.5,
        9.5, 10.5, 15, 6.5, 8.5, 14, 12, 14, 7.5, 16,
        14, 12, 11.5, 11, 11, 9.5, 2, 2, 8, 0,
    ],
    "Mauricio": [
        8, 17, 18, 16, 16, 17, 20, 10, 14, 18,
        16, 22, 21, 14, 18, 18, 11, 21, 19, 16,
        17, 15, 23, 12, 14, 18, 20, 20, 15, 15,
        7, 19, 19, 16, 11, 15, 19, 16, 17, 23,
    ],
}


def validar_dados(dados):
    """Valida participantes, número de posições e medidas antes dos cálculos."""
    integrantes_esperados = {"Lucas C", "Lucas K", "Felipe", "Mauricio"}
    if set(dados) != integrantes_esperados:
        raise ValueError(
            "As chaves de dados_experimento devem ser exatamente: "
            "Lucas C, Lucas K, Felipe e Mauricio."
        )

    for integrante, medidas in dados.items():
        if len(medidas) != NUM_MEDIDAS:
            raise ValueError(
                f"{integrante} tem {len(medidas)} posições; "
                f"eram esperadas exatamente {NUM_MEDIDAS}."
            )
        for indice, medida in enumerate(medidas, start=1):
            if medida is None:
                continue
            if isinstance(medida, bool) or not isinstance(medida, numbers.Real):
                raise ValueError(
                    f"A medida {indice} de {integrante} deve ser um número "
                    "em centímetros ou None."
                )
            if not math.isfinite(float(medida)) or medida < 0:
                raise ValueError(
                    f"A medida {indice} de {integrante} deve ser finita e "
                    "não negativa."
                )


def calcular_estatisticas(df):
    """Calcula resumo descritivo, outliers e aplicações físicas por integrante."""
    resumo = []

    for integrante in df.columns:
        dados = df[integrante].dropna()
        n = len(dados)
        if n == 0:
            continue

        valores = dados.to_numpy(dtype=float)
        media = float(np.mean(valores))
        mediana = float(np.median(valores))
        frequencias = dados.value_counts()
        maior_frequencia = frequencias.max()
        if maior_frequencia == 1:
            moda = "Sem moda"
        else:
            modas = frequencias[frequencias == maior_frequencia].index
            moda = ", ".join(f"{valor:.3f}" for valor in modas)

        desvio_padrao = float(np.std(valores, ddof=1)) if n >= 2 else np.nan
        q1, q2, q3 = np.percentile(valores, [25, 50, 75])
        aiq = q3 - q1
        limite_inferior = q1 - 1.5 * aiq
        limite_superior = q3 + 1.5 * aiq

        medidas = df.index[df[integrante].notna()]
        outliers = [
            f"medida {int(indice) + 1}: {valor:g} cm"
            for indice, valor in zip(medidas, valores)
            if valor < limite_inferior or valor > limite_superior
        ]

        variacao = np.ptp(valores) > 0
        assimetria = (
            float(stats.skew(valores, bias=False)) if n >= 3 and variacao else np.nan
        )
        curtose = (
            float(stats.kurtosis(valores, bias=False, fisher=True))
            if n >= 4 and variacao
            else np.nan
        )

        distancias_m = valores / 100
        tempos_reacao_s = np.sqrt(2 * distancias_m / G_GRAVIDADE)
        tempo_medio_s = float(np.mean(tempos_reacao_s))
        distancia_veiculo_m = VELOCIDADE_MS * tempo_medio_s

        resumo.append(
            {
                "Integrante": integrante,
                "Medidas preenchidas": f"{n}/{NUM_MEDIDAS}",
                "Média (cm)": round(media, 3),
                "Mediana (cm)": round(mediana, 3),
                "Moda(s) (cm)": moda,
                "Desvio padrão amostral (cm)": round(desvio_padrao, 3),
                "Q1 (cm)": round(float(q1), 3),
                "Q2 (cm)": round(float(q2), 3),
                "Q3 (cm)": round(float(q3), 3),
                "AIQ (cm)": round(float(aiq), 3),
                "Limite inferior (cm)": round(float(limite_inferior), 3),
                "Limite superior (cm)": round(float(limite_superior), 3),
                "Outliers": "; ".join(outliers) if outliers else "Nenhum",
                "Assimetria": round(assimetria, 4),
                "Curtose de Fisher": round(curtose, 4),
                "Tempo médio de reação (s)": round(tempo_medio_s, 4),
                "Distância a 100 km/h (m)": round(distancia_veiculo_m, 2),
            }
        )

    return pd.DataFrame(resumo)


def tabela_frequencias(dados, nome_integrante, num_classes=6):
    """Imprime frequências agrupadas; omite a tabela se não houver dados."""
    dados_validos = dados.dropna().to_numpy(dtype=float)
    if len(dados_validos) == 0:
        return

    frequencias, limites = np.histogram(dados_validos, bins=num_classes)
    frequencia_relativa = frequencias / len(dados_validos)
    frequencia_acumulada = np.cumsum(frequencias)
    frequencia_relativa_acumulada = np.cumsum(frequencia_relativa)

    classes = []
    for indice, frequencia in enumerate(frequencias):
        ponto_medio = (limites[indice] + limites[indice + 1]) / 2
        intervalo = f"[{limites[indice]:.2f}, {limites[indice + 1]:.2f}"
        intervalo += "]" if indice == len(frequencias) - 1 else ")"
        classes.append(
            {
                "Intervalo de classes (cm)": intervalo,
                "Ponto médio (xi)": round(ponto_medio, 2),
                "Frequência absoluta (fi)": int(frequencia),
                "Frequência relativa (%)": round(frequencia_relativa[indice] * 100, 2),
                "Frequência acumulada (Fi)": int(frequencia_acumulada[indice]),
                "Frequência relativa acumulada (%)": round(
                    frequencia_relativa_acumulada[indice] * 100, 2
                ),
            }
        )

    print("-" * 80)
    print(f"TABELA DE FREQUÊNCIAS - {nome_integrante}")
    print("-" * 80)
    print(pd.DataFrame(classes).to_string(index=False))
    print()


def gerar_graficos(df):
    """Salva e exibe a evolução temporal, histogramas e boxplots."""
    integrantes_com_dados = [
        integrante for integrante in df.columns if df[integrante].notna().any()
    ]
    if not integrantes_com_dados:
        return

    # Gráfico de linhas: o índice do eixo X preserva a posição original de cada medida.
    plt.figure(figsize=(12, 6))
    for integrante in integrantes_com_dados:
        valores = pd.to_numeric(df[integrante], errors="raise")
        plt.plot(
            np.arange(1, NUM_MEDIDAS + 1),
            valores,
            marker="o",
            label=integrante,
        )
    plt.title("Sequência temporal das medidas (1 a 40)")
    plt.xlabel("Número da medida")
    plt.ylabel("Distância de queda da régua (cm)")
    plt.xticks(range(1, NUM_MEDIDAS + 1, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_sequencia_temporal.png", dpi=300)
    plt.show()

    # Um histograma por integrante, inclusive quando ainda não há medidas preenchidas.
    fig, axes = plt.subplots(
        1, len(df.columns), figsize=(5 * len(df.columns), 4.5), sharey=True
    )
    axes = np.atleast_1d(axes)
    for indice, integrante in enumerate(df.columns):
        valores = df[integrante].dropna()
        if len(valores):
            sns.histplot(valores, kde=False, ax=axes[indice], color="skyblue", bins=6)
        else:
            axes[indice].text(
                0.5, 0.5, "Sem medidas preenchidas", ha="center", va="center"
            )
        axes[indice].set_title(f"Histograma - {integrante}")
        axes[indice].set_xlabel("Distância (cm)")
        axes[indice].set_ylabel("Frequência absoluta")
    fig.suptitle("Distribuição de frequências por integrante", y=1.02)
    fig.tight_layout()
    fig.savefig("grafico_histogramas.png", dpi=300)
    plt.show()

    df_boxplot = df[integrantes_com_dados]
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df_boxplot, palette="Set2")
    plt.title("Comparação dos boxplots dos integrantes")
    plt.ylabel("Distância de queda da régua (cm)")
    plt.tight_layout()
    plt.savefig("grafico_boxplots_comparativo.png", dpi=300)
    plt.show()


def main():
    """Valida a entrada e executa as análises e visualizações do experimento."""
    validar_dados(dados_experimento)
    df_bruto = pd.DataFrame(dados_experimento).apply(pd.to_numeric, errors="raise")

    if not df_bruto.notna().any().any():
        print(
            f"Nenhuma medida preenchida. Insira até {NUM_MEDIDAS} valores em "
            "centímetros para cada integrante em 'dados_experimento' e execute "
            "novamente."
        )
        return

    print("=" * 100)
    print("RESUMO ESTATÍSTICO E APLICAÇÕES FÍSICAS")
    print(f"Gravidade: {G_GRAVIDADE} m/s² | Velocidade do veículo: {VELOCIDADE_KMH:g} km/h")
    print("=" * 100)
    print(calcular_estatisticas(df_bruto).to_string(index=False))
    print()

    for integrante in df_bruto.columns:
        tabela_frequencias(df_bruto[integrante], integrante)

    gerar_graficos(df_bruto)


if __name__ == "__main__":
    main()
