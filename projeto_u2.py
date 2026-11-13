"""Computação Numérica - Projeto da Unidade II: ajuste via RBF e MMQ."""


# ==================== Bibliotecas ====================
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

COR_MEDIDO = "#2a78d6"
COR_MODELO = "#eb6834"
COR_ENTRADA = "#52514e"


# ==================== 1. Leitura dos dados ====================
linhas = []
with open("dados_projeto_02.csv", "r", encoding="utf-8") as arquivo:
    leitor = csv.reader(arquivo)
    cabecalho = next(leitor)
    for linha in leitor:
        if linha:
            linhas.append([float(valor.replace(",", ".")) for valor in linha])

dados = np.array(linhas)
t = dados[:, 0]
u = dados[:, 1]
x = dados[:, 2]
y = dados[:, 3]
N = len(t)

print("Colunas:", cabecalho)
print(f"Número de amostras N = {N}  |  t de {t[0]} a {t[-1]} s  |  Δt = {t[1] - t[0]:.2f} s")
print("Primeiras 3 linhas:\n", dados[:3])


fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Sistema dinâmico a ser identificado", fontweight="bold")

ax[0].plot(t, u, color=COR_ENTRADA, lw=1.5)
ax[0].set_ylabel("Entrada u(t)")

ax[1].plot(t, x, "o", ms=3, alpha=0.35, color=COR_MEDIDO, label="x(t) medido (com ruído)")
ax[1].set_ylabel("Saída 01: x(t)"); ax[1].legend(loc="upper right")

ax[2].plot(t, y, "o", ms=3, alpha=0.35, color=COR_MEDIDO, label="y(t) medido (com ruído)")
ax[2].set_ylabel("Saída 02: y(t)"); ax[2].set_xlabel("Tempo (s)"); ax[2].legend(loc="upper right")

plt.tight_layout()
plt.show()


# ==================== 2. Ferramentas de avaliação (EMQ, Pearson, regressão) ====================
def emq(real, ajustado):
    erro = real - ajustado
    return np.sum(erro ** 2) / len(erro)

def pearson(a, b):
    da = a - np.mean(a)
    db = b - np.mean(b)
    return np.sum(da * db) / np.sqrt(np.sum(da ** 2) * np.sum(db ** 2))

def regressao_linear(xr, yr):
    A = np.array([[len(xr),     np.sum(xr)],
                  [np.sum(xr),  np.sum(xr * xr)]])
    B = np.array([np.sum(yr), np.sum(xr * yr)])
    alfa, beta = np.linalg.solve(A, B)
    return alfa, beta

a_teste, b_teste = np.array([1.0, 2, 3, 4]), np.array([1.1, 1.9, 3.2, 3.9])
print(f"Pearson (nosso) = {pearson(a_teste, b_teste):.6f} | Pearson (np.corrcoef) = {np.corrcoef(a_teste, b_teste)[0, 1]:.6f}")


# ==================== 3. Parte A - RBF: matriz de projeto e ajuste ====================
def matriz_phi(tempos, centros, sigma):
    Phi = np.zeros((len(tempos), len(centros) + 1))
    Phi[:, 0] = 1.0
    for j in range(len(centros)):
        Phi[:, j + 1] = np.exp(-(tempos - centros[j]) ** 2 / (2 * sigma ** 2))
    return Phi

def ajustar_rbf(tempos, sinal, M):
    centros = np.linspace(0, 30, M)
    sigma = centros[1] - centros[0]
    Phi = matriz_phi(tempos, centros, sigma)
    w = np.linalg.solve(Phi.T @ Phi, Phi.T @ sinal)
    return w, centros, sigma


# ==================== 3.4 Escolha do número de gaussianas (validação) ====================
validacao = (np.arange(N) % 5 == 2)
treino = ~validacao
lista_M = np.arange(20, 201, 10)

M_escolhido = {}
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for k, (nome, sinal) in enumerate([("x", x), ("y", y)]):
    erros_val = []
    for M in lista_M:
        w, c, s = ajustar_rbf(t[treino], sinal[treino], M)
        previsto_val = matriz_phi(t[validacao], c, s) @ w
        erros_val.append(emq(sinal[validacao], previsto_val))
    erros_val = np.array(erros_val)

    limite = 1.02 * erros_val.min()
    M_escolhido[nome] = int(lista_M[np.where(erros_val <= limite)[0][0]])

    ax[k].plot(lista_M, erros_val, "o-", color=COR_MODELO, ms=5)
    ax[k].axhline(limite, color="0.5", ls=":", lw=1, label="mínimo + 2%")
    ax[k].axvline(M_escolhido[nome], color="k", ls="--", lw=1, label=f"M escolhido = {M_escolhido[nome]}")
    ax[k].set_title(f"Saída {nome}(t)")
    ax[k].set_xlabel("Número de gaussianas M"); ax[k].set_ylabel("EMQ na validação"); ax[k].legend()
    print(f"{nome}(t): menor EMQ de validação = {erros_val.min():.5f} (M = {lista_M[np.argmin(erros_val)]})"
          f"  ->  M escolhido = {M_escolhido[nome]} (EMQ = {erros_val[lista_M == M_escolhido[nome]][0]:.5f})")
plt.tight_layout()
plt.show()


# ==================== 3.5 Etapa 01 - Ajuste obtido via RBF ====================
w_x, c_x, s_x = ajustar_rbf(t, x, M_escolhido["x"])
w_y, c_y, s_y = ajustar_rbf(t, y, M_escolhido["y"])

x_rbf = matriz_phi(t, c_x, s_x) @ w_x
y_rbf = matriz_phi(t, c_y, s_y) @ w_y

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, ajuste, nome in [(ax[0], x, x_rbf, "x"), (ax[1], y, y_rbf, "y")]:
    a.plot(t, medido, "o", ms=3, alpha=0.35, color=COR_MEDIDO, label="Sinal ruidoso (medido)")
    a.plot(t, ajuste, "--", lw=2, color=COR_MODELO, label="Sinal filtrado / ajustado via RBF")
    a.set_ylabel(f"{nome}(t)"); a.legend(loc="upper right")
ax[0].set_title("Etapa 01 — Ajuste obtido via RBF", fontweight="bold")
ax[1].set_xlabel("Tempo (s)")
plt.tight_layout()
plt.show()


# ==================== Visualização: soma de gaussianas ponderadas ====================
trecho = t <= 4
Phi_y = matriz_phi(t, c_y, s_y)
plt.figure(figsize=(14, 4))
plt.plot(t[trecho], y[trecho], "o", ms=3, alpha=0.35, color=COR_MEDIDO, label="y medido")
for j in range(M_escolhido["y"]):
    if c_y[j] <= 5:
        plt.plot(t[trecho], w_y[j + 1] * Phi_y[trecho, j + 1], color="0.55", lw=1)
plt.plot([], [], color="0.55", lw=1, label="cada sino ponderado: wⱼ·φⱼ(t)")
plt.plot(t[trecho], y_rbf[trecho], color=COR_MODELO, lw=3, label="soma = curva RBF")
plt.title("Como a RBF monta a curva: soma de gaussianas ponderadas")
plt.xlabel("Tempo (s)"); plt.ylabel("y(t)"); plt.legend(loc="lower left")
plt.show()


# ==================== 3.6 Parâmetros livres do modelo RBF ====================
for nome, w, c, s in [("x", w_x, c_x, s_x), ("y", w_y, c_y, s_y)]:
    print(f"Modelo RBF para {nome}(t):")
    print(f"  M = {len(c)} gaussianas | centros de {c[0]:.2f} a {c[-1]:.2f} s | σ = {s:.4f} s")
    print(f"  total de pesos ajustados = {len(w)} (w0 + {len(c)})")
    print(f"  w0 = {w[0]:.4f} | primeiros pesos w1..w5 = {np.round(w[1:6], 4)}")
    print()


# ==================== 3.7 Etapa 02 - EMQ do ajuste RBF ====================
EMQ_x_rbf = emq(x, x_rbf)
EMQ_y_rbf = emq(y, y_rbf)

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, ajuste, nome, e in [(ax[0], x, x_rbf, "x", EMQ_x_rbf), (ax[1], y, y_rbf, "y", EMQ_y_rbf)]:
    a.plot(t, medido, "o", ms=3, alpha=0.35, color=COR_MEDIDO, label="Dados")
    a.plot(t, ajuste, lw=2, color=COR_MODELO, label="Ajuste RBF")
    a.text(0.01, 0.95, f"EMQ = {e:.4f}", transform=a.transAxes, va="top",
           bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    a.set_ylabel(f"{nome}(t) / {nome}(t) filtrado"); a.legend(loc="upper right")
ax[0].set_title("Etapa 02 — EMQ do ajuste via RBF", fontweight="bold")
ax[1].set_xlabel("Tempo (s)")
plt.tight_layout()
plt.show()

print(f"EMQ x(t) = {EMQ_x_rbf:.5f}   ->  erro típico √EMQ = {np.sqrt(EMQ_x_rbf):.4f}")
print(f"EMQ y(t) = {EMQ_y_rbf:.5f}   ->  erro típico √EMQ = {np.sqrt(EMQ_y_rbf):.4f}")


# ==================== 3.8 Etapa 03 - Regressão linear e Pearson (RBF) ====================
def grafico_regressao(ax, ajustado, real, nome, cor_faixa, rotulo_modelo):
    alfa, beta = regressao_linear(ajustado, real)
    r = pearson(real, ajustado)
    e = emq(real, ajustado)
    xs = np.linspace(ajustado.min(), ajustado.max(), 100)
    reta = alfa + beta * xs
    ax.plot(ajustado, real, "o", ms=4, alpha=0.3, color=COR_MEDIDO)
    ax.plot(xs, reta, "k--", lw=1.5, label=f"Regressão linear: real = {alfa:.4f} + {beta:.4f}·ajustado")
    ax.fill_between(xs, reta - np.sqrt(e), reta + np.sqrt(e), color=cor_faixa, alpha=0.2, label="± √EMQ")
    ax.text(0.01, 0.95, f"EMQ = {e:.4f}\nr = {r:.4f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    ax.set_xlabel(f"{nome}(t) ajustado ({rotulo_modelo})"); ax.set_ylabel(f"{nome}(t) medido")
    ax.legend(loc="lower right")
    return alfa, beta, r

fig, ax = plt.subplots(2, 1, figsize=(14, 9))
alfa_x_rbf, beta_x_rbf, r_x_rbf = grafico_regressao(ax[0], x_rbf, x, "x", COR_MODELO, "RBF")
alfa_y_rbf, beta_y_rbf, r_y_rbf = grafico_regressao(ax[1], y_rbf, y, "y", COR_MODELO, "RBF")
ax[0].set_title("Etapa 03 — Regressão linear (medido × ajustado RBF) e correlação de Pearson", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"x(t): reta real = {alfa_x_rbf:.4f} + {beta_x_rbf:.4f}·ajustado | Pearson r = {r_x_rbf:.4f} | r² = {r_x_rbf**2:.4f}")
print(f"y(t): reta real = {alfa_y_rbf:.4f} + {beta_y_rbf:.4f}·ajustado | Pearson r = {r_y_rbf:.4f} | r² = {r_y_rbf**2:.4f}")


# ==================== 4. Parte B - MMQ: modelo dinâmico ====================
def montar_regressores(s, u):
    A = np.column_stack((s[1:-1], s[:-2], u[2:], u[1:-1]))
    b = s[2:]
    return A, b

def ajustar_mmq(s, u):
    A, b = montar_regressores(s, u)
    theta = np.linalg.solve(A.T @ A, A.T @ b)
    return theta

theta_x = ajustar_mmq(x, u)
theta_y = ajustar_mmq(y, u)

print("Parâmetros livres do modelo y[n] = a1·y[n-1] + a2·y[n-2] + b0·u[n] + b1·u[n-1]\n")
print(f"{'saída':<8}{'a1':>10}{'a2':>10}{'b0':>10}{'b1':>10}")
for nome, th in [("x", theta_x), ("y", theta_y)]:
    print(f"{nome:<8}{th[0]:>10.4f}{th[1]:>10.4f}{th[2]:>10.4f}{th[3]:>10.4f}")


# ==================== 4.3 Etapa 04 - Ajuste via MMQ ====================
A_x, alvo_x = montar_regressores(x, u)
A_y, alvo_y = montar_regressores(y, u)
x_mmq = A_x @ theta_x
y_mmq = A_y @ theta_y
t_mmq = t[2:]

EMQ_x_mmq = emq(alvo_x, x_mmq)
EMQ_y_mmq = emq(alvo_y, y_mmq)

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, ajuste, nome, e in [(ax[0], alvo_x, x_mmq, "x", EMQ_x_mmq), (ax[1], alvo_y, y_mmq, "y", EMQ_y_mmq)]:
    a.plot(t_mmq, medido, "o", ms=3, alpha=0.35, color=COR_MEDIDO, label=f"{nome}(t) + ruído (medido)")
    a.plot(t_mmq, ajuste, lw=1, color=COR_MODELO, label=f"{nome}(t) ajustado via MMQ")
    a.text(0.01, 0.95, f"EMQ = {e:.4f}", transform=a.transAxes, va="top",
           bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    a.set_ylabel(f"{nome}(t)"); a.legend(loc="upper right")
ax[0].set_title("Etapa 04 — Ajuste via MMQ (modelo dinâmico, predição um passo à frente)", fontweight="bold")
ax[1].set_xlabel("Tempo (s)")
plt.tight_layout()
plt.show()

print(f"EMQ x(t) via MMQ = {EMQ_x_mmq:.5f}")
print(f"EMQ y(t) via MMQ = {EMQ_y_mmq:.5f}")


# ==================== 4.4 Regressão linear e Pearson (MMQ) ====================
fig, ax = plt.subplots(2, 1, figsize=(14, 9))
alfa_x_mmq, beta_x_mmq, r_x_mmq = grafico_regressao(ax[0], x_mmq, alvo_x, "x", "tab:red", "MMQ")
alfa_y_mmq, beta_y_mmq, r_y_mmq = grafico_regressao(ax[1], y_mmq, alvo_y, "y", "#eda100", "MMQ")
ax[0].set_title("Regressão linear (medido × ajustado MMQ) e correlação de Pearson", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"x(t): reta real = {alfa_x_mmq:.4f} + {beta_x_mmq:.4f}·ajustado | Pearson r = {r_x_mmq:.4f} | r² = {r_x_mmq**2:.4f}")
print(f"y(t): reta real = {alfa_y_mmq:.4f} + {beta_y_mmq:.4f}·ajustado | Pearson r = {r_y_mmq:.4f} | r² = {r_y_mmq**2:.4f}")


# ==================== 4.5 Simulação livre do modelo MMQ ====================
def simular_livre(theta, u, s_inicial):
    a1, a2, b0, b1 = theta
    s_sim = np.zeros(len(u))
    s_sim[0], s_sim[1] = s_inicial[0], s_inicial[1]
    for n in range(2, len(u)):
        s_sim[n] = a1 * s_sim[n - 1] + a2 * s_sim[n - 2] + b0 * u[n] + b1 * u[n - 1]
    return s_sim

x_sim = simular_livre(theta_x, u, x)
y_sim = simular_livre(theta_y, u, y)

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, rbf, sim, nome in [(ax[0], x, x_rbf, x_sim, "x"), (ax[1], y, y_rbf, y_sim, "y")]:
    a.plot(t, medido, "o", ms=3, alpha=0.25, color=COR_MEDIDO, label="medido")
    a.plot(t, rbf, color="k", lw=1.5, ls="--", label="filtrado via RBF (referência)")
    a.plot(t, sim, color=COR_MODELO, lw=2, label="simulação livre do modelo MMQ")
    a.set_ylabel(f"{nome}(t)"); a.legend(loc="upper right")
ax[0].set_title("Simulação livre do modelo MMQ (usa só a entrada u)", fontweight="bold")
ax[1].set_xlabel("Tempo (s)")
plt.tight_layout()
plt.show()

for nome, medido, sim in [("x", x, x_sim), ("y", y, y_sim)]:
    print(f"{nome}(t) simulação livre: EMQ = {emq(medido[2:], sim[2:]):.5f} | Pearson r = {pearson(medido[2:], sim[2:]):.4f}")


# ==================== 5. Comparação final ====================
print(f"{'Modelo':<34}{'Saída':<7}{'EMQ':>10}{'√EMQ':>9}{'Pearson r':>11}{'r²':>8}")
print("-" * 79)
linhas_tabela = [
    ("RBF (filtragem no tempo)",           "x", emq(x, x_rbf),          pearson(x, x_rbf)),
    ("RBF (filtragem no tempo)",           "y", emq(y, y_rbf),          pearson(y, y_rbf)),
    ("MMQ - predição um passo à frente",   "x", EMQ_x_mmq,              r_x_mmq),
    ("MMQ - predição um passo à frente",   "y", EMQ_y_mmq,              r_y_mmq),
    ("MMQ - simulação livre",              "x", emq(x[2:], x_sim[2:]),  pearson(x[2:], x_sim[2:])),
    ("MMQ - simulação livre",              "y", emq(y[2:], y_sim[2:]),  pearson(y[2:], y_sim[2:])),
]
for modelo, saida, e, r in linhas_tabela:
    print(f"{modelo:<34}{saida:<7}{e:>10.5f}{np.sqrt(e):>9.4f}{r:>11.4f}{r**2:>8.4f}")

print(f"\nNúmero de parâmetros: RBF x = {len(w_x)}, RBF y = {len(w_y)}, MMQ = 4 por saída")
