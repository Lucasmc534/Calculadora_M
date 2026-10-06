"""
Calculadora estatística do experimento de queda livre da régua.

CONTEXTO E REQUISITOS
---------------------
Processa até 40 medidas de distância (cm) para cada participante, preservando
a ordem original. None representa dado ausente; zero é mantido como observação.
Valida as quatro listas antes de calcular estatísticas, frequência, quartis,
outliers, assimetria, curtose e aplicações físicas.

As medidas são convertidas para metros e o tempo de queda é estimado por
t = sqrt(2*d/g), supondo queda a partir do repouso e sem resistência do ar,
com g = 9,80665 m/s². O deslocamento do veículo usa v = 100/3,6 m/s vezes a
média dos tempos individuais.

Quartis usam interpolação linear (mesma convenção de numpy.percentile), o
desvio padrão é amostral, a assimetria e o excesso de curtose usam correções
amostrais (curtose de Fisher). As seis classes de frequência têm largura
igual. Outliers são definidos pelas cercas de Tukey: Q1 - 1,5*AIQ e
Q3 + 1,5*AIQ.

GRÁFICOS LOCAIS
---------------
O programa usa somente a biblioteca padrão do Python: não requer NumPy,
Pandas, SciPy, Matplotlib, Seaborn nem DLLs de terceiros. Gera imagens SVG
autônomas, visualizáveis localmente em um navegador, e um HTML de índice:
  - grafico_sequencia_temporal.svg
  - histograma_<participante>.svg (um por participante)
  - boxplot_<participante>.svg (um por participante)
  - grafico_boxplots_comparativo.svg
  - graficos_calculadora.html

Todos os arquivos de saída são gravados ao lado deste script. Para trocar ou
corrigir dados, edite as listas em dados_experimento; mantenha 40 posições.
Para uma medida não coletada, use None sem remover sua posição da lista.

O valor vazio que originalmente apareceu na lista de Mauricio depois do 14
foi tratado como erro de digitação, conforme confirmação do usuário; os 40
valores numéricos foram mantidos em ordem.
"""

import html
import math
import numbers
import re
from pathlib import Path


NUM_MEDIDAS = 40
NUM_CLASSES = 6
G_GRAVIDADE = 9.80665
VELOCIDADE_KMH = 100.0
VELOCIDADE_MS = VELOCIDADE_KMH / 3.6

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

CORES = ("#2563eb", "#dc2626", "#059669", "#9333ea")


def validar_dados(dados):
    """Exige quatro participantes, 40 posições e medidas finitas não negativas."""
    esperados = {"Lucas C", "Lucas K", "Felipe", "Mauricio"}
    if set(dados) != esperados:
        raise ValueError(
            "dados_experimento deve conter exatamente Lucas C, Lucas K, "
            "Felipe e Mauricio."
        )

    for pessoa, medidas in dados.items():
        if len(medidas) != NUM_MEDIDAS:
            raise ValueError(
                f"{pessoa} tem {len(medidas)} posições; "
                f"eram esperadas {NUM_MEDIDAS}."
            )
        for numero, valor in enumerate(medidas, start=1):
            if valor is None:
                continue
            if isinstance(valor, bool) or not isinstance(valor, numbers.Real):
                raise ValueError(
                    f"A medida {numero} de {pessoa} precisa ser um número ou None."
                )
            if not math.isfinite(float(valor)) or valor < 0:
                raise ValueError(
                    f"A medida {numero} de {pessoa} precisa ser finita "
                    "e não negativa."
                )


def quantil(valores, proporcao):
    """Calcula quantil com interpolação linear entre observações ordenadas."""
    ordenados = sorted(valores)
    if not ordenados:
        raise ValueError("Não é possível calcular quantil sem observações.")
    posicao = (len(ordenados) - 1) * proporcao
    inferior = math.floor(posicao)
    superior = math.ceil(posicao)
    if inferior == superior:
        return ordenados[inferior]
    fracao = posicao - inferior
    return ordenados[inferior] * (1 - fracao) + ordenados[superior] * fracao


def calcular_moda(valores):
    """Retorna todas as modas; quando cada observação é única, não há moda."""
    contagens = {}
    for valor in valores:
        contagens[valor] = contagens.get(valor, 0) + 1
    maxima = max(contagens.values())
    if maxima == 1:
        return "Sem moda"
    return ", ".join(
        f"{valor:g}" for valor, contagem in sorted(contagens.items())
        if contagem == maxima
    )


def calcular_assimetria(valores):
    """Coeficiente de Fisher-Pearson corrigido para amostra."""
    n = len(valores)
    if n < 3:
        return math.nan
    media = sum(valores) / n
    m2 = sum((valor - media) ** 2 for valor in valores) / n
    if m2 == 0:
        return math.nan
    m3 = sum((valor - media) ** 3 for valor in valores) / n
    g1 = m3 / (m2 ** 1.5)
    return math.sqrt(n * (n - 1)) / (n - 2) * g1


def calcular_curtose_fisher(valores):
    """Excesso de curtose de Fisher com correção para amostra."""
    n = len(valores)
    if n < 4:
        return math.nan
    media = sum(valores) / n
    m2 = sum((valor - media) ** 2 for valor in valores) / n
    if m2 == 0:
        return math.nan
    m4 = sum((valor - media) ** 4 for valor in valores) / n
    g2 = m4 / (m2 * m2) - 3
    return ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6)


def frequencias_agrupadas(valores, numero_classes=NUM_CLASSES):
    """Agrupa observações em classes iguais; inclui o máximo na última classe."""
    minimo = min(valores)
    maximo = max(valores)
    if minimo == maximo:
        margem = abs(minimo) * 0.05 or 0.5
        minimo -= margem
        maximo += margem
    largura = (maximo - minimo) / numero_classes
    frequencias = [0] * numero_classes
    for valor in valores:
        indice = int((valor - minimo) / largura)
        indice = min(max(indice, 0), numero_classes - 1)
        frequencias[indice] += 1
    limites = [minimo + largura * indice for indice in range(numero_classes + 1)]
    return limites, frequencias


def calcular_resumo(dados):
    """Calcula estatísticas de cada pessoa, mantendo os índices das observações."""
    resumo = {}
    for pessoa, medidas in dados.items():
        observacoes = [
            (indice, float(valor))
            for indice, valor in enumerate(medidas, start=1)
            if valor is not None
        ]
        valores = [valor for _, valor in observacoes]
        if not valores:
            continue

        n = len(valores)
        media = sum(valores) / n
        mediana = quantil(valores, 0.5)
        q1, q3 = quantil(valores, 0.25), quantil(valores, 0.75)
        aiq = q3 - q1
        limite_inferior = q1 - 1.5 * aiq
        limite_superior = q3 + 1.5 * aiq
        outliers = [
            (indice, valor)
            for indice, valor in observacoes
            if valor < limite_inferior or valor > limite_superior
        ]
        variancia_amostral = (
            sum((valor - media) ** 2 for valor in valores) / (n - 1)
            if n >= 2 else math.nan
        )
        tempos = [math.sqrt(2 * (valor / 100) / G_GRAVIDADE) for valor in valores]
        tempo_medio = sum(tempos) / n

        resumo[pessoa] = {
            "observacoes": observacoes,
            "valores": valores,
            "n": n,
            "media": media,
            "mediana": mediana,
            "moda": calcular_moda(valores),
            "desvio_padrao": math.sqrt(variancia_amostral),
            "q1": q1,
            "q2": mediana,
            "q3": q3,
            "aiq": aiq,
            "limite_inferior": limite_inferior,
            "limite_superior": limite_superior,
            "outliers": outliers,
            "assimetria": calcular_assimetria(valores),
            "curtose": calcular_curtose_fisher(valores),
            "tempo_medio": tempo_medio,
            "distancia_veiculo": VELOCIDADE_MS * tempo_medio,
        }
    return resumo


def formatar_numero(valor, casas=3):
    return "NaN" if math.isnan(valor) else f"{valor:.{casas}f}"


def imprimir_resumo(resumo):
    print("=" * 105)
    print("RESUMO ESTATÍSTICO E APLICAÇÕES FÍSICAS")
    print(
        f"Gravidade: {G_GRAVIDADE} m/s² | "
        f"Velocidade: {VELOCIDADE_KMH:g} km/h ({VELOCIDADE_MS:.4f} m/s)"
    )
    print("=" * 105)
    for pessoa, item in resumo.items():
        outliers = (
            "; ".join(f"medida {indice}: {valor:g} cm"
                      for indice, valor in item["outliers"])
            or "Nenhum"
        )
        print(f"\n{pessoa} — {item['n']}/{NUM_MEDIDAS} medidas")
        print(f"  Média: {item['media']:.3f} cm | Mediana: {item['mediana']:.3f} cm")
        print(f"  Moda(s): {item['moda']} cm")
        print(f"  Desvio padrão amostral: {formatar_numero(item['desvio_padrao'])} cm")
        print(
            f"  Q1: {item['q1']:.3f} | Q2: {item['q2']:.3f} | "
            f"Q3: {item['q3']:.3f} | AIQ: {item['aiq']:.3f} cm"
        )
        print(
            f"  Limites de outlier: {item['limite_inferior']:.3f} a "
            f"{item['limite_superior']:.3f} cm | Outliers: {outliers}"
        )
        print(
            f"  Assimetria: {formatar_numero(item['assimetria'], 4)} | "
            f"Curtose de Fisher: {formatar_numero(item['curtose'], 4)}"
        )
        print(
            f"  Tempo médio: {item['tempo_medio']:.4f} s | "
            f"Distância a 100 km/h: {item['distancia_veiculo']:.2f} m"
        )


def imprimir_tabelas_frequencia(resumo):
    for pessoa, item in resumo.items():
        limites, frequencias = frequencias_agrupadas(item["valores"])
        total = item["n"]
        acumulada = 0
        print(f"\nTABELA DE FREQUÊNCIAS — {pessoa}")
        print(
            f"{'Intervalo (cm)':<24}{'Ponto médio':>13}"
            f"{'fi':>7}{'fr (%)':>10}{'Fi':>8}{'Fr (%)':>10}"
        )
        for indice, frequencia in enumerate(frequencias):
            acumulada += frequencia
            inicio, fim = limites[indice], limites[indice + 1]
            fechamento = "]" if indice == len(frequencias) - 1 else ")"
            intervalo = f"[{inicio:.2f}, {fim:.2f}{fechamento}"
            ponto_medio = (inicio + fim) / 2
            print(
                f"{intervalo:<24}{ponto_medio:>13.2f}{frequencia:>7}"
                f"{frequencia / total * 100:>10.2f}"
                f"{acumulada:>8}{acumulada / total * 100:>10.2f}"
            )


def svg_inicio(titulo, largura, altura):
    titulo_xml = html.escape(titulo)
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" '
        f'height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{largura / 2}" y="32" text-anchor="middle" '
        f'font-family="Arial,sans-serif" font-size="20" font-weight="bold" '
        f'fill="#172033">{titulo_xml}</text>',
    ]


def svg_texto(x, y, texto, tamanho=12, ancora="middle", cor="#334155"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{ancora}" '
        f'font-family="Arial,sans-serif" font-size="{tamanho}" fill="{cor}">'
        f'{html.escape(str(texto))}</text>'
    )


def salvar_svg(nome, linhas):
    destino = Path(__file__).resolve().parent / nome
    destino.write_text("\n".join(linhas + ["</svg>"]), encoding="utf-8")
    return destino


def desenhar_eixos(linhas, x, y, largura, altura, x_min, x_max, y_min, y_max,
                   rotulo_x, rotulo_y):
    linhas.append(
        f'<path d="M{x},{y} V{y + altura} H{x + largura}" '
        'fill="none" stroke="#475569" stroke-width="1.5"/>'
    )
    for passo in range(6):
        valor_y = y_min + (y_max - y_min) * passo / 5
        py = y + altura - altura * passo / 5
        linhas.append(
            f'<path d="M{x},{py:.2f} H{x + largura}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        linhas.append(svg_texto(x - 10, py + 4, f"{valor_y:g}", 10, "end"))
        valor_x = x_min + (x_max - x_min) * passo / 5
        px = x + largura * passo / 5
        linhas.append(svg_texto(px, y + altura + 20, f"{valor_x:g}", 10))
    linhas.append(svg_texto(x + largura / 2, y + altura + 48, rotulo_x, 12))
    linhas.append(
        f'<text x="20" y="{y + altura / 2}" text-anchor="middle" '
        'font-family="Arial,sans-serif" font-size="12" fill="#334155" '
        f'transform="rotate(-90 20 {y + altura / 2})">'
        f'{html.escape(rotulo_y)}</text>'
    )


def gerar_grafico_linhas(resumo):
    largura_svg, altura_svg = 1000, 520
    margem_x, margem_y = 80, 75
    grafico_w, grafico_h = 850, 350
    valores = [
        valor for item in resumo.values() for _, valor in item["observacoes"]
    ]
    y_min = min(0.0, min(valores))
    y_max = max(valores)
    if y_min == y_max:
        y_max = y_min + 1
    linhas = svg_inicio("Evolução das medidas por integrante", largura_svg, altura_svg)
    desenhar_eixos(
        linhas, margem_x, margem_y, grafico_w, grafico_h, 1, NUM_MEDIDAS,
        y_min, y_max, "Número da medida", "Distância (cm)"
    )

    for serie, (pessoa, item) in enumerate(resumo.items()):
        cor = CORES[serie % len(CORES)]
        pontos = item["observacoes"]
        for (indice_a, valor_a), (indice_b, valor_b) in zip(pontos, pontos[1:]):
            if indice_b != indice_a + 1:
                continue
            x1 = margem_x + (indice_a - 1) / (NUM_MEDIDAS - 1) * grafico_w
            x2 = margem_x + (indice_b - 1) / (NUM_MEDIDAS - 1) * grafico_w
            y1 = margem_y + grafico_h - (valor_a - y_min) / (y_max - y_min) * grafico_h
            y2 = margem_y + grafico_h - (valor_b - y_min) / (y_max - y_min) * grafico_h
            linhas.append(
                f'<path d="M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}" '
                f'fill="none" stroke="{cor}" stroke-width="2"/>'
            )
        for indice, valor in pontos:
            px = margem_x + (indice - 1) / (NUM_MEDIDAS - 1) * grafico_w
            py = margem_y + grafico_h - (valor - y_min) / (y_max - y_min) * grafico_h
            linhas.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{cor}"/>')
        legend_x = margem_x + serie * 190
        linhas.append(
            f'<path d="M{legend_x},{altura_svg - 22} h24" '
            f'stroke="{cor}" stroke-width="3"/>'
        )
        linhas.append(svg_texto(legend_x + 31, altura_svg - 18, pessoa, 12, "start"))
    return salvar_svg("grafico_sequencia_temporal.svg", linhas)


def gerar_histograma(pessoa, item, cor):
    largura_svg, altura_svg = 680, 440
    margem_x, margem_y = 75, 65
    grafico_w, grafico_h = 540, 285
    limites, frequencias = frequencias_agrupadas(item["valores"])
    max_frequencia = max(frequencias) or 1
    linhas = svg_inicio(f"Histograma — {pessoa}", largura_svg, altura_svg)
    linhas.append(
        f'<path d="M{margem_x},{margem_y} V{margem_y + grafico_h} '
        f'H{margem_x + grafico_w}" fill="none" stroke="#475569" stroke-width="1.5"/>'
    )
    for passo in range(6):
        valor = max_frequencia * passo / 5
        py = margem_y + grafico_h - grafico_h * passo / 5
        linhas.append(
            f'<path d="M{margem_x},{py:.2f} H{margem_x + grafico_w}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        linhas.append(svg_texto(margem_x - 10, py + 4, f"{valor:.1f}", 10, "end"))
    largura_barra = grafico_w / len(frequencias)
    for indice, frequencia in enumerate(frequencias):
        altura = grafico_h * frequencia / max_frequencia
        x = margem_x + indice * largura_barra + 1
        y = margem_y + grafico_h - altura
        linhas.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{largura_barra - 2:.2f}" '
            f'height="{altura:.2f}" fill="{cor}" fill-opacity="0.78" '
            'stroke="#ffffff"/>'
        )
        linhas.append(svg_texto(x + largura_barra / 2, y - 6, frequencia, 10))
        linhas.append(
            svg_texto(x + largura_barra / 2, margem_y + grafico_h + 19,
                      f"{limites[indice]:.1f}–{limites[indice + 1]:.1f}", 9)
        )
    linhas.append(
        svg_texto(margem_x + grafico_w / 2, margem_y + grafico_h + 50,
                  "Distância (cm)", 12)
    )
    linhas.append(
        f'<text x="20" y="{margem_y + grafico_h / 2}" text-anchor="middle" '
        'font-family="Arial,sans-serif" font-size="12" fill="#334155" '
        f'transform="rotate(-90 20 {margem_y + grafico_h / 2})">'
        'Frequência absoluta</text>'
    )
    return salvar_svg(f"histograma_{slug(pessoa)}.svg", linhas)


def desenhar_box(linhas, pessoa, item, cor, centro_x, y_topo, escala,
                 mostrar_rotulo=True):
    q1, mediana, q3 = item["q1"], item["mediana"], item["q3"]
    inferior, superior = item["limite_inferior"], item["limite_superior"]
    valores_dentro = [v for v in item["valores"] if inferior <= v <= superior]
    bigode_min, bigode_max = min(valores_dentro), max(valores_dentro)
    y = lambda valor: y_topo - valor * escala
    largura_caixa = 48
    linhas.append(
        f'<path d="M{centro_x},{y(bigode_max):.2f} V{y(q3):.2f} '
        f'M{centro_x - 12},{y(bigode_max):.2f} H{centro_x + 12} '
        f'M{centro_x},{y(q1):.2f} V{y(bigode_min):.2f} '
        f'M{centro_x - 12},{y(bigode_min):.2f} H{centro_x + 12}" '
        'fill="none" stroke="#334155" stroke-width="2"/>'
    )
    linhas.append(
        f'<rect x="{centro_x - largura_caixa / 2}" y="{y(q3):.2f}" '
        f'width="{largura_caixa}" height="{max(1, (q3 - q1) * escala):.2f}" '
        f'fill="{cor}" fill-opacity="0.35" stroke="{cor}" stroke-width="2"/>'
    )
    linhas.append(
        f'<path d="M{centro_x - largura_caixa / 2},{y(mediana):.2f} '
        f'H{centro_x + largura_caixa / 2}" stroke="{cor}" stroke-width="3"/>'
    )
    for indice, valor in enumerate(item["valores"]):
        if valor < inferior or valor > superior:
            deslocamento = ((indice % 5) - 2) * 4
            linhas.append(
                f'<circle cx="{centro_x + deslocamento}" cy="{y(valor):.2f}" '
                'r="4" fill="#dc2626" stroke="#ffffff"/>'
            )
    if mostrar_rotulo:
        linhas.append(svg_texto(centro_x, y_topo + 24, pessoa, 12))


def gerar_boxplot(pessoa, item, cor):
    largura_svg, altura_svg = 360, 440
    y_topo = 340
    maximo = max(item["valores"])
    escala = 260 / max(maximo, 1)
    linhas = svg_inicio(f"Boxplot — {pessoa}", largura_svg, altura_svg)
    linhas.append(
        f'<path d="M65,{y_topo} V70" stroke="#cbd5e1" stroke-width="1"/>'
    )
    desenhar_box(linhas, pessoa, item, cor, 180, y_topo, escala, False)
    for valor, nome in (
        (item["q1"], "Q1"), (item["mediana"], "Mediana"), (item["q3"], "Q3")
    ):
        linhas.append(svg_texto(250, y_topo - valor * escala + 4,
                                f"{nome}: {valor:.2f} cm", 11, "start"))
    linhas.append(svg_texto(180, 400, "Distância (cm)", 12))
    return salvar_svg(f"boxplot_{slug(pessoa)}.svg", linhas)


def gerar_boxplots_comparativos(resumo):
    largura_svg, altura_svg = 760, 460
    y_topo = 350
    maior_valor = max(max(item["valores"]) for item in resumo.values())
    escala = 260 / max(maior_valor, 1)
    linhas = svg_inicio("Boxplots comparativos dos integrantes", largura_svg, altura_svg)
    linhas.append(f'<path d="M70,{y_topo} V70 H720" fill="none" stroke="#475569"/>')
    for passo in range(6):
        valor = maior_valor * passo / 5
        py = y_topo - valor * escala
        linhas.append(f'<path d="M70,{py:.2f} H720" stroke="#e2e8f0"/>')
        linhas.append(svg_texto(62, py + 4, f"{valor:.1f}", 10, "end"))
    passo_x = 640 / max(len(resumo), 1)
    for indice, (pessoa, item) in enumerate(resumo.items()):
        centro = 70 + passo_x * (indice + 0.5)
        desenhar_box(
            linhas, pessoa, item, CORES[indice % len(CORES)],
            centro, y_topo, escala
        )
    linhas.append(svg_texto(395, 420, "Distância (cm)", 12))
    return salvar_svg("grafico_boxplots_comparativo.svg", linhas)


def slug(texto):
    sem_acentos = texto.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_-]", "", sem_acentos)


def gerar_indice_html(arquivos):
    cards = "\n".join(
        f'<section><h2>{html.escape(titulo)}</h2>'
        f'<a href="{html.escape(nome, quote=True)}">'
        f'<img src="{html.escape(nome, quote=True)}" '
        f'alt="{html.escape(titulo, quote=True)}"></a></section>'
        for titulo, nome in arquivos
    )
    conteudo = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gráficos da calculadora</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #172033; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 1rem; }}
section {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; }}
img {{ width: 100%; height: auto; }}
</style>
</head>
<body><h1>Gráficos do experimento</h1><main>{cards}</main></body>
</html>
"""
    destino = Path(__file__).resolve().parent / "graficos_calculadora.html"
    destino.write_text(conteudo, encoding="utf-8")
    return destino


def gerar_graficos(resumo):
    """Gera SVGs independentes e um índice HTML; não abre janelas gráficas."""
    arquivos = []
    arquivos.append(("Evolução temporal", gerar_grafico_linhas(resumo).name))
    for indice, (pessoa, item) in enumerate(resumo.items()):
        cor = CORES[indice % len(CORES)]
        arquivos.append((f"Histograma — {pessoa}", gerar_histograma(pessoa, item, cor).name))
        arquivos.append((f"Boxplot — {pessoa}", gerar_boxplot(pessoa, item, cor).name))
    arquivos.append(
        ("Boxplots comparativos", gerar_boxplots_comparativos(resumo).name)
    )
    arquivos.append(("Índice dos gráficos", gerar_indice_html(arquivos).name))
    return arquivos


def main():
    validar_dados(dados_experimento)
    resumo = calcular_resumo(dados_experimento)
    if not resumo:
        raise ValueError("Nenhuma medida foi informada para análise.")
    imprimir_resumo(resumo)
    imprimir_tabelas_frequencia(resumo)
    arquivos = gerar_graficos(resumo)
    pasta = Path(__file__).resolve().parent
    print("\nArquivos de gráficos gerados:")
    for titulo, nome in arquivos:
        print(f"  {titulo}: {pasta / nome}")
    print("\nAbra graficos_calculadora.html para visualizar os gráficos juntos.")


if __name__ == "__main__":
    main()
