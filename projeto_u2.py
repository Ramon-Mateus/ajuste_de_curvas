# ==================== Bibliotecas ====================
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5})
COR_MEDIDO = "#2a78d6"   # azul: dados medidos
COR_RBF = "#eb6834"      # laranja: modelo RBF
COR_MMQ = "#6250d6"      # roxo: modelo MMQ


# ==================== 1. Leitura dos dados ====================
linhas = []
with open("dados_projeto_02.csv", "r", encoding="utf-8") as arquivo:
    leitor = csv.reader(arquivo)
    cabecalho = next(leitor)              # primeira linha: nomes das colunas
    for linha in leitor:
        if linha:                         # ignora linhas vazias
            linhas.append([float(valor.replace(",", ".")) for valor in linha])

dados = np.array(linhas)
t = dados[:, 0]   # tempo
u = dados[:, 1]   # entrada
x = dados[:, 2]   # saída 01
y = dados[:, 3]   # saída 02
N = len(t)

print("Colunas:", cabecalho)
print(f"N = {N} amostras | t de {t[0]} a {t[-1]} s | Δt = {t[1] - t[0]:.2f} s")
print(f"x varia de {x.min():.2f} a {x.max():.2f} | y varia de {y.min():.2f} a {y.max():.2f} | u assume {np.unique(u)}")

fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
ax[0].plot(t, u, color="0.3", lw=1.5)
ax[0].set_ylabel("entrada u(t)")
ax[1].plot(t, x, "o", ms=3, alpha=0.35, color=COR_MEDIDO)
ax[1].set_ylabel("saída x(t)")
ax[2].plot(t, y, "o", ms=3, alpha=0.35, color=COR_MEDIDO)
ax[2].set_ylabel("saída y(t)")
ax[2].set_xlabel("tempo (s)")
ax[0].set_title("Dados a serem ajustados", fontweight="bold")
plt.tight_layout()
plt.show()


# ==================== 2. Ferramentas: EMQ, Pearson, regressão linear e equação normal ====================
def emq(real, ajustado):
    erro = real - ajustado
    return np.sum(erro ** 2) / len(erro)

def pearson(a, b):
    da = a - np.mean(a)
    db = b - np.mean(b)
    return np.sum(da * db) / np.sqrt(np.sum(da ** 2) * np.sum(db ** 2))

def regressao_linear(xr, yr):
    A = np.array([[len(xr),    np.sum(xr)],
                  [np.sum(xr), np.sum(xr * xr)]])
    B = np.array([np.sum(yr), np.sum(xr * yr)])
    alfa, beta = np.linalg.solve(A, B)
    return alfa, beta

def equacao_normal(A, alvo):
    return np.linalg.solve(A.T @ A, A.T @ alvo)

# Conferência: o nosso Pearson tem de bater com o do NumPy
a_teste = np.array([1.0, 2.0, 3.0, 4.0])
b_teste = np.array([1.1, 1.9, 3.2, 3.9])
print(f"Pearson (nosso) = {pearson(a_teste, b_teste):.6f} | Pearson (NumPy) = {np.corrcoef(a_teste, b_teste)[0, 1]:.6f}")


# ==================== 3. Montando as entradas do modelo (regressores) ====================
def montar_regressores(s, u):
    Z = np.column_stack((s[1:-1], s[:-2], u[2:], u[1:-1]))   # [s[n-1], s[n-2], u[n], u[n-1]]
    alvo = s[2:]                                             # s[n]
    return Z, alvo

Z_x, alvo_x = montar_regressores(x, u)
Z_y, alvo_y = montar_regressores(y, u)
t_alvo = t[2:]   # tempo de cada linha

print(f"Z_x tem {Z_x.shape[0]} linhas e {Z_x.shape[1]} colunas")
print(f"Primeira linha (n = 2): z = {np.round(Z_x[0], 4)}  ->  alvo x[2] = {alvo_x[0]:.4f}")
print(f"Conferindo: x[1] = {x[1]:.4f}, x[0] = {x[0]:.4f}, u[2] = {u[2]:.0f}, u[1] = {u[1]:.0f}")


# ==================== 4.1/4.2 RBF: matriz das gaussianas e escolha dos centros ====================
def matriz_phi(Z, centros, sigma):
    Phi = np.ones((len(Z), len(centros) + 1))              # coluna 0 = 1 (para w0)
    for j in range(len(centros)):
        distancia2 = np.sum((Z - centros[j]) ** 2, axis=1)   # ||z - c_j||² para todas as linhas
        Phi[:, j + 1] = np.exp(-distancia2 / (2 * sigma ** 2))
    return Phi

def escolher_centros(Z, por_estado, por_troca=3):
    centros = []
    for u_atual, u_anterior, quantidade in [(1, 1, por_estado), (-1, -1, por_estado),
                                            (1, -1, por_troca), (-1, 1, por_troca)]:
        linhas = np.where((Z[:, 2] == u_atual) & (Z[:, 3] == u_anterior))[0]          # linhas desta combinação
        linhas = linhas[np.argsort(Z[linhas, 0])]                                      # ordena por s[n-1]
        escolhidas = linhas[np.linspace(0, len(linhas) - 1, quantidade).astype(int)]  # igualmente espaçadas
        centros.append(Z[escolhidas])
    return np.vstack(centros)

for u_atual, u_anterior in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
    quantidade = np.sum((Z_x[:, 2] == u_atual) & (Z_x[:, 3] == u_anterior))
    print(f"(u[n], u[n-1]) = ({u_atual:+d}, {u_anterior:+d}): {quantidade} linhas")


# ==================== 4.3 RBF: quantos centros e qual sigma (validação) ====================
gerador = np.random.default_rng(0)
validacao = gerador.random(len(alvo_x)) < 0.2   # True em ~20% das linhas
treino = ~validacao

lista_por_estado = [5, 10, 15, 20]
lista_sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75]

escolha = {}
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for k, (nome, Z, alvo) in enumerate([("x", Z_x, alvo_x), ("y", Z_y, alvo_y)]):
    tabela = np.zeros((len(lista_por_estado), len(lista_sigma)))
    for i, por_estado in enumerate(lista_por_estado):
        centros = escolher_centros(Z[treino], por_estado)
        for j, sigma in enumerate(lista_sigma):
            w = equacao_normal(matriz_phi(Z[treino], centros, sigma), alvo[treino])
            tabela[i, j] = emq(alvo[validacao], matriz_phi(Z[validacao], centros, sigma) @ w)

    limite = 1.02 * tabela.min()
    i, j = np.argwhere(tabela <= limite)[0]          # 1º que passa: menos centros, depois menor sigma
    escolha[nome] = (lista_por_estado[i], lista_sigma[j])

    print(f"EMQ de validação de {nome} (linhas: centros por estado | colunas: sigma = {lista_sigma})")
    for linha_i, por_estado in enumerate(lista_por_estado):
        print(f"  {por_estado:>2} por estado (M = {2 * por_estado + 6:>2}): {np.round(tabela[linha_i], 5)}")
    print(f"  -> escolhido: {escolha[nome][0]} por estado (M = {2 * escolha[nome][0] + 6}), sigma = {escolha[nome][1]}\n")

    for linha_i, por_estado in enumerate(lista_por_estado):
        ax[k].plot(lista_sigma, tabela[linha_i], "o-", ms=4, label=f"M = {2 * por_estado + 6}")
    ax[k].axhline(limite, color="0.5", ls=":", label="mínimo + 2%")
    ax[k].set_title(f"{nome}: EMQ de validação")
    ax[k].set_xlabel("σ")
    ax[k].set_ylabel("EMQ na validação")
    ax[k].set_ylim(tabela.min() * 0.98, tabela.min() * 1.25)
    ax[k].legend(fontsize=8)
plt.tight_layout()
plt.show()


# ==================== 4.4 RBF: ajuste final e parâmetros livres ====================
modelo_rbf = {}
for nome, Z, alvo in [("x", Z_x, alvo_x), ("y", Z_y, alvo_y)]:
    por_estado, sigma = escolha[nome]
    centros = escolher_centros(Z, por_estado)
    Phi = matriz_phi(Z, centros, sigma)
    w = equacao_normal(Phi, alvo)
    modelo_rbf[nome] = (centros, sigma, w)

    print(f"RBF de {nome}: M = {len(centros)} gaussianas | σ = {sigma} | {len(w)} pesos (w0 + {len(centros)})"
          f" | matriz Φ: {Phi.shape[0]} x {Phi.shape[1]}")
    print(f"  w0 = {w[0]:.4f}")
    print("  alguns centros [s[n-1], s[n-2], u[n], u[n-1]] e seus pesos:")
    for j in [0, 1, len(centros) - 6, len(centros) - 1]:
        print(f"    c{j + 1:<2} = {np.round(centros[j], 3)}   w{j + 1} = {w[j + 1]:.4f}")

x_rbf = matriz_phi(Z_x, modelo_rbf["x"][0], modelo_rbf["x"][1]) @ modelo_rbf["x"][2]
y_rbf = matriz_phi(Z_y, modelo_rbf["y"][0], modelo_rbf["y"][1]) @ modelo_rbf["y"][2]

# Gráfico: onde estão os centros (2 das 4 coordenadas; a cor indica u[n])
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for a, Z, nome in [(ax[0], Z_x, "x"), (ax[1], Z_y, "y")]:
    centros = modelo_rbf[nome][0]
    for valor_u, cor in [(1, COR_MEDIDO), (-1, "#1baf7a")]:
        sel = Z[:, 2] == valor_u
        a.plot(Z[sel, 0], Z[sel, 1], "o", ms=3, alpha=0.25, color=cor, label=f"dados com u[n] = {valor_u:+d}")
        selc = centros[:, 2] == valor_u
        a.plot(centros[selc, 0], centros[selc, 1], "^", ms=10, color=cor, markeredgecolor="k",
               label=f"centros com u[n] = {valor_u:+d}")
    a.set_xlabel(f"{nome}[n-1]")
    a.set_ylabel(f"{nome}[n-2]")
    a.set_title(f"{nome}: os {len(centros)} centros estão sobre os dados")
    a.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()


# ==================== 4.5 Etapas 01 e 02 - Ajuste via RBF e EMQ ====================
EMQ_x_rbf = emq(alvo_x, x_rbf)
EMQ_y_rbf = emq(alvo_y, y_rbf)

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, ajuste, nome, e in [(ax[0], alvo_x, x_rbf, "x", EMQ_x_rbf), (ax[1], alvo_y, y_rbf, "y", EMQ_y_rbf)]:
    a.plot(t_alvo, medido, "o", ms=3, alpha=0.3, color=COR_MEDIDO, label="medido (com ruído)")
    a.plot(t_alvo, ajuste, color=COR_RBF, lw=1.2, label=f"{nome}[n] ajustado via RBF")
    a.text(0.01, 0.95, f"EMQ = {e:.4f}", transform=a.transAxes, va="top",
           bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    a.set_ylabel(f"{nome}(t)")
    a.legend(loc="upper right", fontsize=9)
ax[0].set_title("Etapas 01 e 02 — Ajuste via RBF e EMQ", fontweight="bold")
ax[1].set_xlabel("tempo (s)")
plt.tight_layout()
plt.show()

print(f"EMQ x (RBF) = {EMQ_x_rbf:.5f}  ->  erro típico √EMQ = {np.sqrt(EMQ_x_rbf):.4f}")
print(f"EMQ y (RBF) = {EMQ_y_rbf:.5f}  ->  erro típico √EMQ = {np.sqrt(EMQ_y_rbf):.4f}")


# ==================== 4.6 Etapa 03 - Regressão linear (real x ajustado) e Pearson da RBF ====================
def grafico_regressao(ax, ajustado, real, nome, cor, rotulo_modelo):
    alfa, beta = regressao_linear(ajustado, real)
    r = pearson(real, ajustado)
    e = emq(real, ajustado)
    xs = np.linspace(ajustado.min(), ajustado.max(), 100)
    reta = alfa + beta * xs
    ax.plot(ajustado, real, "o", ms=4, alpha=0.3, color=COR_MEDIDO)
    ax.plot(xs, reta, "k--", lw=1.5, label=f"regressão: real = {alfa:.4f} + {beta:.4f}·ajustado")
    ax.fill_between(xs, reta - np.sqrt(e), reta + np.sqrt(e), color=cor, alpha=0.2, label="± √EMQ")
    ax.text(0.01, 0.95, f"EMQ = {e:.4f}\nr = {r:.4f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    ax.set_xlabel(f"{nome}[n] ajustado ({rotulo_modelo})")
    ax.set_ylabel(f"{nome}[n] medido")
    ax.legend(loc="lower right")
    return alfa, beta, r

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
alfa_x_rbf, beta_x_rbf, r_x_rbf = grafico_regressao(ax[0], x_rbf, alvo_x, "x", COR_RBF, "RBF")
alfa_y_rbf, beta_y_rbf, r_y_rbf = grafico_regressao(ax[1], y_rbf, alvo_y, "y", COR_RBF, "RBF")
fig.suptitle("Etapa 03 — Regressão linear (medido × ajustado RBF) e Pearson", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"x: real = {alfa_x_rbf:.4f} + {beta_x_rbf:.4f}·ajustado | Pearson r = {r_x_rbf:.4f} | r² = {r_x_rbf**2:.4f}")
print(f"y: real = {alfa_y_rbf:.4f} + {beta_y_rbf:.4f}·ajustado | Pearson r = {r_y_rbf:.4f} | r² = {r_y_rbf**2:.4f}")


# ==================== 5.1 Etapa 04 - MMQ: parâmetros livres, ajuste e EMQ ====================
theta_x = equacao_normal(Z_x, alvo_x)
theta_y = equacao_normal(Z_y, alvo_y)

print("Parâmetros livres de s[n] = a1·s[n-1] + a2·s[n-2] + b0·u[n] + b1·u[n-1]\n")
print(f"{'saída':<8}{'a1':>10}{'a2':>10}{'b0':>10}{'b1':>10}")
for nome, th in [("x", theta_x), ("y", theta_y)]:
    print(f"{nome:<8}{th[0]:>10.4f}{th[1]:>10.4f}{th[2]:>10.4f}{th[3]:>10.4f}")

x_mmq = Z_x @ theta_x
y_mmq = Z_y @ theta_y
EMQ_x_mmq = emq(alvo_x, x_mmq)
EMQ_y_mmq = emq(alvo_y, y_mmq)

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, ajuste, nome, e in [(ax[0], alvo_x, x_mmq, "x", EMQ_x_mmq), (ax[1], alvo_y, y_mmq, "y", EMQ_y_mmq)]:
    a.plot(t_alvo, medido, "o", ms=3, alpha=0.3, color=COR_MEDIDO, label="medido (com ruído)")
    a.plot(t_alvo, ajuste, color=COR_MMQ, lw=1.2, label=f"{nome}[n] ajustado via MMQ")
    a.text(0.01, 0.95, f"EMQ = {e:.4f}", transform=a.transAxes, va="top",
           bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    a.set_ylabel(f"{nome}(t)")
    a.legend(loc="upper right", fontsize=9)
ax[0].set_title("Etapa 04 — Ajuste via MMQ e EMQ", fontweight="bold")
ax[1].set_xlabel("tempo (s)")
plt.tight_layout()
plt.show()

print(f"\nEMQ x (MMQ) = {EMQ_x_mmq:.5f}  ->  erro típico √EMQ = {np.sqrt(EMQ_x_mmq):.4f}")
print(f"EMQ y (MMQ) = {EMQ_y_mmq:.5f}  ->  erro típico √EMQ = {np.sqrt(EMQ_y_mmq):.4f}")


# ==================== 5.2 Etapa 04 - Regressão linear e Pearson do MMQ ====================
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
alfa_x_mmq, beta_x_mmq, r_x_mmq = grafico_regressao(ax[0], x_mmq, alvo_x, "x", COR_MMQ, "MMQ")
alfa_y_mmq, beta_y_mmq, r_y_mmq = grafico_regressao(ax[1], y_mmq, alvo_y, "y", COR_MMQ, "MMQ")
fig.suptitle("Etapa 04 — Regressão linear (medido × ajustado MMQ) e Pearson", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"x: real = {alfa_x_mmq:.4f} + {beta_x_mmq:.4f}·ajustado | Pearson r = {r_x_mmq:.4f} | r² = {r_x_mmq**2:.4f}")
print(f"y: real = {alfa_y_mmq:.4f} + {beta_y_mmq:.4f}·ajustado | Pearson r = {r_y_mmq:.4f} | r² = {r_y_mmq**2:.4f}")


# ==================== 6. Simulação livre (os modelos recebem só a entrada u) ====================
def simular_livre(prever, u, s_medido):
    s_sim = np.zeros(len(u))
    s_sim[0], s_sim[1] = s_medido[0], s_medido[1]           # só as 2 primeiras amostras vêm dos dados
    for n in range(2, len(u)):
        z = np.array([[s_sim[n - 1], s_sim[n - 2], u[n], u[n - 1]]])
        s_sim[n] = prever(z)[0]
    return s_sim

def prever_rbf(nome):
    centros, sigma, w = modelo_rbf[nome]
    return lambda z: matriz_phi(z, centros, sigma) @ w

def prever_mmq(theta):
    return lambda z: z @ theta

x_sim_rbf = simular_livre(prever_rbf("x"), u, x)
y_sim_rbf = simular_livre(prever_rbf("y"), u, y)
x_sim_mmq = simular_livre(prever_mmq(theta_x), u, x)
y_sim_mmq = simular_livre(prever_mmq(theta_y), u, y)

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, s_rbf, s_mmq, nome in [(ax[0], x, x_sim_rbf, x_sim_mmq, "x"), (ax[1], y, y_sim_rbf, y_sim_mmq, "y")]:
    a.plot(t, medido, "o", ms=3, alpha=0.2, color=COR_MEDIDO, label="medido")
    a.plot(t, s_rbf, color=COR_RBF, lw=2, label="simulação livre — RBF")
    a.plot(t, s_mmq, color=COR_MMQ, lw=2, label="simulação livre — MMQ")
    a.set_ylabel(f"{nome}(t)")
    a.legend(loc="upper right", fontsize=9)
ax[0].set_title("Simulação livre (os modelos recebem só a entrada u)", fontweight="bold")
ax[1].set_xlabel("tempo (s)")
plt.tight_layout()
plt.show()

for nome, medido, s_rbf, s_mmq in [("x", x, x_sim_rbf, x_sim_mmq), ("y", y, y_sim_rbf, y_sim_mmq)]:
    print(f"{nome}: RBF livre EMQ = {emq(medido[2:], s_rbf[2:]):.5f}, r = {pearson(medido[2:], s_rbf[2:]):.4f}"
          f"  |  MMQ livre EMQ = {emq(medido[2:], s_mmq[2:]):.5f}, r = {pearson(medido[2:], s_mmq[2:]):.4f}")


# ==================== 7. Comparação final ====================
print(f"{'Modelo':<34}{'Saída':<7}{'EMQ':>10}{'√EMQ':>9}{'Pearson r':>11}{'r²':>8}")
print("-" * 79)
linhas_tabela = [
    ("RBF - um passo à frente", "x", EMQ_x_rbf, r_x_rbf),
    ("RBF - um passo à frente", "y", EMQ_y_rbf, r_y_rbf),
    ("MMQ - um passo à frente", "x", EMQ_x_mmq, r_x_mmq),
    ("MMQ - um passo à frente", "y", EMQ_y_mmq, r_y_mmq),
    ("RBF - simulação livre",   "x", emq(x[2:], x_sim_rbf[2:]), pearson(x[2:], x_sim_rbf[2:])),
    ("RBF - simulação livre",   "y", emq(y[2:], y_sim_rbf[2:]), pearson(y[2:], y_sim_rbf[2:])),
    ("MMQ - simulação livre",   "x", emq(x[2:], x_sim_mmq[2:]), pearson(x[2:], x_sim_mmq[2:])),
    ("MMQ - simulação livre",   "y", emq(y[2:], y_sim_mmq[2:]), pearson(y[2:], y_sim_mmq[2:])),
]
for modelo, saida, e, r in linhas_tabela:
    print(f"{modelo:<34}{saida:<7}{e:>10.5f}{np.sqrt(e):>9.4f}{r:>11.4f}{r**2:>8.4f}")

print(f"\nParâmetros: RBF x = {len(modelo_rbf['x'][2])} pesos, RBF y = {len(modelo_rbf['y'][2])} pesos | MMQ = 4 por saída")