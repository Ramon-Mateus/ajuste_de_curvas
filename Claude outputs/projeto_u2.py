"""Computação Numérica - Projeto da Unidade II: ajuste via RBF e MMQ."""


# ==================== Bibliotecas ====================
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

COR_MEDIDO = "#2a78d6"
COR_RBF = "#eb6834"
COR_MMQ = "#6250d6"
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


# ==================== 2. Ferramentas: EMQ, Pearson, regressão, matriz RBF, equação normal ====================
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

def matriz_phi(Z, centros, sigma):
    Phi = np.ones((len(Z), len(centros) + 1))
    for j in range(len(centros)):
        distancia2 = np.sum((Z - centros[j]) ** 2, axis=1)
        Phi[:, j + 1] = np.exp(-distancia2 / (2 * sigma ** 2))
    return Phi

def equacao_normal(A, alvo):
    return np.linalg.solve(A.T @ A, A.T @ alvo)

a_teste, b_teste = np.array([1.0, 2, 3, 4]), np.array([1.1, 1.9, 3.2, 3.9])
print(f"Pearson (nosso) = {pearson(a_teste, b_teste):.6f} | Pearson (np.corrcoef) = {np.corrcoef(a_teste, b_teste)[0, 1]:.6f}")


# ==================== 3. Preparação: RBF com uma entrada (tempo) - escolha de M e filtragem ====================
def ajustar_rbf_tempo(tempos, sinal, M):
    centros = np.linspace(0, 30, M)
    sigma = centros[1] - centros[0]
    Phi = matriz_phi(tempos.reshape(-1, 1), centros.reshape(-1, 1), sigma)
    return equacao_normal(Phi, sinal), centros, sigma

def prever_rbf_tempo(tempos, w, centros, sigma):
    return matriz_phi(tempos.reshape(-1, 1), centros.reshape(-1, 1), sigma) @ w

validacao_t = (np.arange(N) % 5 == 2)
treino_t = ~validacao_t
lista_M = np.arange(20, 201, 10)

M_tempo = {}
fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
for k, (nome, sinal) in enumerate([("x", x), ("y", y)]):
    erros_val = []
    for M in lista_M:
        w, c, s = ajustar_rbf_tempo(t[treino_t], sinal[treino_t], M)
        erros_val.append(emq(sinal[validacao_t], prever_rbf_tempo(t[validacao_t], w, c, s)))
    erros_val = np.array(erros_val)
    limite = 1.02 * erros_val.min()
    M_tempo[nome] = int(lista_M[np.where(erros_val <= limite)[0][0]])
    ax[k].plot(lista_M, erros_val, "o-", color=COR_RBF, ms=4)
    ax[k].axvline(M_tempo[nome], color="k", ls="--", lw=1, label=f"M escolhido = {M_tempo[nome]}")
    ax[k].set_title(f"{nome}(t): EMQ de validação × número de gaussianas")
    ax[k].set_xlabel("M"); ax[k].set_ylabel("EMQ na validação"); ax[k].legend()
plt.tight_layout()
plt.show()

w_xt, c_xt, s_xt = ajustar_rbf_tempo(t, x, M_tempo["x"])
w_yt, c_yt, s_yt = ajustar_rbf_tempo(t, y, M_tempo["y"])
x_filtrado = prever_rbf_tempo(t, w_xt, c_xt, s_xt)
y_filtrado = prever_rbf_tempo(t, w_yt, c_yt, s_yt)

for nome, M, c, s, sinal, filt in [("x", M_tempo["x"], c_xt, s_xt, x, x_filtrado),
                                   ("y", M_tempo["y"], c_yt, s_yt, y, y_filtrado)]:
    print(f"{nome}(t): M = {M} gaussianas, centros de {c[0]:.0f} a {c[-1]:.0f} s a cada {s:.3f} s (σ = {s:.3f} s)"
          f" | EMQ = {emq(sinal, filt):.5f} | r = {pearson(sinal, filt):.4f}")


fig, ax = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 0.9]})
for a, medido, filt, c, nome in [(ax[0], x, x_filtrado, c_xt, "x"), (ax[1], y, y_filtrado, c_yt, "y")]:
    a.plot(t, medido, "o", ms=3, alpha=0.3, color=COR_MEDIDO, label="medido (com ruído)")
    a.plot(t, filt, color="k", lw=2, label="filtrado via RBF no tempo (referência)")
    a.plot(c, np.full_like(c, medido.min() - 0.05), "^", color=COR_RBF, ms=5, label="centros das gaussianas")
    a.set_ylabel(f"{nome}(t)"); a.legend(loc="upper right", fontsize=9)
ax[0].set_title("RBF com uma entrada (o tempo): filtragem dos sinais", fontweight="bold")

trecho = t <= 4
Phi_y = matriz_phi(t.reshape(-1, 1), c_yt.reshape(-1, 1), s_yt)
ax[2].plot(t[trecho], y[trecho], "o", ms=3, alpha=0.35, color=COR_MEDIDO, label="y medido")
for j in range(len(c_yt)):
    if c_yt[j] <= 5:
        ax[2].plot(t[trecho], w_yt[j + 1] * Phi_y[trecho, j + 1], color="0.6", lw=1)
ax[2].plot([], [], color="0.6", lw=1, label="cada gaussiana × seu peso: wⱼ·φⱼ(t)")
ax[2].plot(t[trecho], y_filtrado[trecho], color="k", lw=2.5, label="soma de todas = curva RBF")
ax[2].set_title("Como a RBF monta a curva: soma de gaussianas ponderadas (y, de 0 a 4 s)")
ax[2].set_xlabel("Tempo (s)"); ax[2].set_ylabel("y(t)"); ax[2].legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.show()


# ==================== 4.1/4.2 RBF dinâmica: regressores e escolha dos centros ====================
def montar_regressores(s, u):
    Z = np.column_stack((s[1:-1], s[:-2], u[2:], u[1:-1]))
    alvo = s[2:]
    return Z, alvo

def escolher_centros(Z, por_estado=10, por_troca=3):
    centros = []
    for u_atual, u_anterior, quantidade in [(1, 1, por_estado), (-1, -1, por_estado),
                                            (1, -1, por_troca), (-1, 1, por_troca)]:
        linhas = np.where((Z[:, 2] == u_atual) & (Z[:, 3] == u_anterior))[0]
        linhas = linhas[np.argsort(Z[linhas, 0])]
        escolhidas = linhas[np.linspace(0, len(linhas) - 1, quantidade).astype(int)]
        centros.append(Z[escolhidas])
    return np.vstack(centros)

Z_x, alvo_x = montar_regressores(x, u)
Z_y, alvo_y = montar_regressores(y, u)
t_alvo = t[2:]

for u_atual, u_anterior in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
    qtd = np.sum((Z_x[:, 2] == u_atual) & (Z_x[:, 3] == u_anterior))
    print(f"(u[n], u[n-1]) = ({u_atual:+d}, {u_anterior:+d}): {qtd} linhas")
print(f"Total: {len(Z_x)} linhas (n = 2 ... {N - 1})")


# ==================== 4.3 RBF dinâmica: escolha de sigma (validação) ====================
gerador = np.random.default_rng(0)
validacao_z = gerador.random(len(alvo_x)) < 0.2
treino_z = ~validacao_z
lista_sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75]

sigma_rbf = {}
fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
for k, (nome, Z, alvo) in enumerate([("x", Z_x, alvo_x), ("y", Z_y, alvo_y)]):
    centros_tr = escolher_centros(Z[treino_z])
    erros_val = []
    for sigma in lista_sigma:
        w = equacao_normal(matriz_phi(Z[treino_z], centros_tr, sigma), alvo[treino_z])
        erros_val.append(emq(alvo[validacao_z], matriz_phi(Z[validacao_z], centros_tr, sigma) @ w))
    erros_val = np.array(erros_val)
    limite = 1.02 * erros_val.min()
    sigma_rbf[nome] = lista_sigma[np.where(erros_val <= limite)[0][0]]
    ax[k].plot(lista_sigma, erros_val, "o-", color=COR_RBF, ms=5)
    ax[k].axvline(sigma_rbf[nome], color="k", ls="--", lw=1, label=f"σ escolhido = {sigma_rbf[nome]}")
    ax[k].set_title(f"{nome}: EMQ de validação × σ (26 centros)")
    ax[k].set_xlabel("σ"); ax[k].set_ylabel("EMQ na validação"); ax[k].legend()
    print(f"{nome}: EMQ de validação por σ = {np.round(erros_val, 5)} -> σ escolhido = {sigma_rbf[nome]}")
plt.tight_layout()
plt.show()


# ==================== 4.4 RBF dinâmica: ajuste final e parâmetros livres ====================
centros_x = escolher_centros(Z_x)
centros_y = escolher_centros(Z_y)
Phi_x = matriz_phi(Z_x, centros_x, sigma_rbf["x"])
Phi_y = matriz_phi(Z_y, centros_y, sigma_rbf["y"])
w_rbf_x = equacao_normal(Phi_x, alvo_x)
w_rbf_y = equacao_normal(Phi_y, alvo_y)

x_rbf = Phi_x @ w_rbf_x
y_rbf = Phi_y @ w_rbf_y

for nome, C, s, w, Phi in [("x", centros_x, sigma_rbf["x"], w_rbf_x, Phi_x),
                           ("y", centros_y, sigma_rbf["y"], w_rbf_y, Phi_y)]:
    print(f"Modelo RBF para {nome}[n]:  M = {len(C)} centros | σ = {s} | {len(w)} pesos (w0 + {len(C)})"
          f" | matriz Φ: {Phi.shape[0]} x {Phi.shape[1]}")
    print(f"  w0 = {w[0]:.4f}")
    print("  primeiros centros  [s[n-1], s[n-2], u[n], u[n-1]]  e seus pesos:")
    for j in [0, 1, 2, 10, 20, 23]:
        print(f"    c{j + 1:<2} = {np.round(C[j], 3)}   w{j + 1} = {w[j + 1]:.4f}")


# ==================== 4.4 Onde estão os centros (gráfico) ====================
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for a, Z, C, nome in [(ax[0], Z_x, centros_x, "x"), (ax[1], Z_y, centros_y, "y")]:
    for valor_u, cor in [(1, COR_MEDIDO), (-1, "#1baf7a")]:
        sel = Z[:, 2] == valor_u
        a.plot(Z[sel, 0], Z[sel, 1], "o", ms=3, alpha=0.25, color=cor, label=f"dados com u[n] = {valor_u:+d}")
        selc = C[:, 2] == valor_u
        a.plot(C[selc, 0], C[selc, 1], "^", ms=10, color=cor, markeredgecolor="k", label=f"centros com u[n] = {valor_u:+d}")
    a.set_xlabel(f"{nome}[n-1]"); a.set_ylabel(f"{nome}[n-2]")
    a.set_title(f"{nome}: dados e os {len(C)} centros no plano ({nome}[n-1], {nome}[n-2])")
    a.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()


# ==================== 4.5 Etapas 01 e 02 - Ajuste via RBF e EMQ ====================
EMQ_x_rbf = emq(alvo_x, x_rbf)
EMQ_y_rbf = emq(alvo_y, y_rbf)

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, ajuste, filt, nome, e in [(ax[0], alvo_x, x_rbf, x_filtrado[2:], "x", EMQ_x_rbf),
                                         (ax[1], alvo_y, y_rbf, y_filtrado[2:], "y", EMQ_y_rbf)]:
    a.plot(t_alvo, medido, "o", ms=3, alpha=0.3, color=COR_MEDIDO, label="medido (com ruído)")
    a.plot(t_alvo, filt, "k--", lw=1.5, label="referência filtrada (Seção 3)")
    a.plot(t_alvo, ajuste, color=COR_RBF, lw=1.2, label=f"{nome}[n] ajustado via RBF")
    a.text(0.01, 0.95, f"EMQ = {e:.4f}", transform=a.transAxes, va="top",
           bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    a.set_ylabel(f"{nome}(t)"); a.legend(loc="upper right", fontsize=9)
ax[0].set_title("Etapas 01 e 02 — Ajuste via RBF de s[n] = f(s[n-1], s[n-2], u[n], u[n-1]) e EMQ", fontweight="bold")
ax[1].set_xlabel("Tempo (s)")
plt.tight_layout()
plt.show()

print(f"EMQ x (RBF) = {EMQ_x_rbf:.5f}   ->  erro típico √EMQ = {np.sqrt(EMQ_x_rbf):.4f}")
print(f"EMQ y (RBF) = {EMQ_y_rbf:.5f}   ->  erro típico √EMQ = {np.sqrt(EMQ_y_rbf):.4f}")


# ==================== 4.6 Etapa 03 - Regressão linear e Pearson (RBF) ====================
def grafico_regressao(ax, ajustado, real, nome, cor, rotulo_modelo):
    alfa, beta = regressao_linear(ajustado, real)
    r = pearson(real, ajustado)
    e = emq(real, ajustado)
    xs = np.linspace(ajustado.min(), ajustado.max(), 100)
    reta = alfa + beta * xs
    ax.plot(ajustado, real, "o", ms=4, alpha=0.3, color=COR_MEDIDO)
    ax.plot(xs, reta, "k--", lw=1.5, label=f"Regressão linear: real = {alfa:.4f} + {beta:.4f}·ajustado")
    ax.fill_between(xs, reta - np.sqrt(e), reta + np.sqrt(e), color=cor, alpha=0.2, label="± √EMQ")
    ax.text(0.01, 0.95, f"EMQ = {e:.4f}\nr = {r:.4f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    ax.set_xlabel(f"{nome}[n] ajustado ({rotulo_modelo})"); ax.set_ylabel(f"{nome}[n] medido")
    ax.legend(loc="lower right")
    return alfa, beta, r

fig, ax = plt.subplots(2, 1, figsize=(14, 9))
alfa_x_rbf, beta_x_rbf, r_x_rbf = grafico_regressao(ax[0], x_rbf, alvo_x, "x", COR_RBF, "RBF")
alfa_y_rbf, beta_y_rbf, r_y_rbf = grafico_regressao(ax[1], y_rbf, alvo_y, "y", COR_RBF, "RBF")
ax[0].set_title("Etapa 03 — Regressão linear (medido × ajustado RBF) e correlação de Pearson", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"x: reta real = {alfa_x_rbf:.4f} + {beta_x_rbf:.4f}·ajustado | Pearson r = {r_x_rbf:.4f} | r² = {r_x_rbf**2:.4f}")
print(f"y: reta real = {alfa_y_rbf:.4f} + {beta_y_rbf:.4f}·ajustado | Pearson r = {r_y_rbf:.4f} | r² = {r_y_rbf**2:.4f}")


# ==================== 5. Etapa 04 - MMQ: parâmetros, ajuste e EMQ ====================
theta_x = equacao_normal(Z_x, alvo_x)
theta_y = equacao_normal(Z_y, alvo_y)

print("Parâmetros livres do modelo s[n] = a1·s[n-1] + a2·s[n-2] + b0·u[n] + b1·u[n-1]\n")
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
    a.set_ylabel(f"{nome}(t)"); a.legend(loc="upper right", fontsize=9)
ax[0].set_title("Etapa 04 — Ajuste via MMQ (predição um passo à frente) e EMQ", fontweight="bold")
ax[1].set_xlabel("Tempo (s)")
plt.tight_layout()
plt.show()

print(f"\nEMQ x (MMQ) = {EMQ_x_mmq:.5f}")
print(f"EMQ y (MMQ) = {EMQ_y_mmq:.5f}")


# ==================== 5.3 Etapa 04 - Regressão linear e Pearson (MMQ) ====================
fig, ax = plt.subplots(2, 1, figsize=(14, 9))
alfa_x_mmq, beta_x_mmq, r_x_mmq = grafico_regressao(ax[0], x_mmq, alvo_x, "x", COR_MMQ, "MMQ")
alfa_y_mmq, beta_y_mmq, r_y_mmq = grafico_regressao(ax[1], y_mmq, alvo_y, "y", COR_MMQ, "MMQ")
ax[0].set_title("Etapa 04 — Regressão linear (medido × ajustado MMQ) e correlação de Pearson", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"x: reta real = {alfa_x_mmq:.4f} + {beta_x_mmq:.4f}·ajustado | Pearson r = {r_x_mmq:.4f} | r² = {r_x_mmq**2:.4f}")
print(f"y: reta real = {alfa_y_mmq:.4f} + {beta_y_mmq:.4f}·ajustado | Pearson r = {r_y_mmq:.4f} | r² = {r_y_mmq**2:.4f}")


# ==================== 6. Simulação livre (RBF e MMQ) ====================
def simular_livre(modelo, u, s_inicial):
    s_sim = np.zeros(len(u))
    s_sim[0], s_sim[1] = s_inicial[0], s_inicial[1]
    for n in range(2, len(u)):
        z = np.array([s_sim[n - 1], s_sim[n - 2], u[n], u[n - 1]])
        s_sim[n] = modelo(z)
    return s_sim

modelo_rbf_x = lambda z: (matriz_phi(z.reshape(1, -1), centros_x, sigma_rbf["x"]) @ w_rbf_x)[0]
modelo_rbf_y = lambda z: (matriz_phi(z.reshape(1, -1), centros_y, sigma_rbf["y"]) @ w_rbf_y)[0]
modelo_mmq_x = lambda z: z @ theta_x
modelo_mmq_y = lambda z: z @ theta_y

x_sim_rbf = simular_livre(modelo_rbf_x, u, x)
y_sim_rbf = simular_livre(modelo_rbf_y, u, y)
x_sim_mmq = simular_livre(modelo_mmq_x, u, x)
y_sim_mmq = simular_livre(modelo_mmq_y, u, y)

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, medido, filt, s_rbf, s_mmq, nome in [(ax[0], x, x_filtrado, x_sim_rbf, x_sim_mmq, "x"),
                                            (ax[1], y, y_filtrado, y_sim_rbf, y_sim_mmq, "y")]:
    a.plot(t, medido, "o", ms=3, alpha=0.2, color=COR_MEDIDO, label="medido")
    a.plot(t, filt, "k--", lw=1.5, label="referência filtrada (Seção 3)")
    a.plot(t, s_rbf, color=COR_RBF, lw=2, label="simulação livre — RBF")
    a.plot(t, s_mmq, color=COR_MMQ, lw=2, label="simulação livre — MMQ")
    a.set_ylabel(f"{nome}(t)"); a.legend(loc="upper right", fontsize=9)
ax[0].set_title("Simulação livre (os modelos recebem só a entrada u)", fontweight="bold")
ax[1].set_xlabel("Tempo (s)")
plt.tight_layout()
plt.show()

for nome, medido, s_rbf, s_mmq in [("x", x, x_sim_rbf, x_sim_mmq), ("y", y, y_sim_rbf, y_sim_mmq)]:
    print(f"{nome}: RBF livre  EMQ = {emq(medido[2:], s_rbf[2:]):.5f}, r = {pearson(medido[2:], s_rbf[2:]):.4f}"
          f"  |  MMQ livre  EMQ = {emq(medido[2:], s_mmq[2:]):.5f}, r = {pearson(medido[2:], s_mmq[2:]):.4f}")


# ==================== 7. Comparação final ====================
print(f"{'Modelo':<38}{'Saída':<7}{'EMQ':>10}{'√EMQ':>9}{'Pearson r':>11}{'r²':>8}")
print("-" * 83)
linhas_tabela = [
    ("RBF dinâmica - um passo à frente",   "x", EMQ_x_rbf, r_x_rbf),
    ("RBF dinâmica - um passo à frente",   "y", EMQ_y_rbf, r_y_rbf),
    ("MMQ - um passo à frente",            "x", EMQ_x_mmq, r_x_mmq),
    ("MMQ - um passo à frente",            "y", EMQ_y_mmq, r_y_mmq),
    ("RBF dinâmica - simulação livre",     "x", emq(x[2:], x_sim_rbf[2:]), pearson(x[2:], x_sim_rbf[2:])),
    ("RBF dinâmica - simulação livre",     "y", emq(y[2:], y_sim_rbf[2:]), pearson(y[2:], y_sim_rbf[2:])),
    ("MMQ - simulação livre",              "x", emq(x[2:], x_sim_mmq[2:]), pearson(x[2:], x_sim_mmq[2:])),
    ("MMQ - simulação livre",              "y", emq(y[2:], y_sim_mmq[2:]), pearson(y[2:], y_sim_mmq[2:])),
    ("Referência: RBF no tempo (filtro)",  "x", emq(x, x_filtrado), pearson(x, x_filtrado)),
    ("Referência: RBF no tempo (filtro)",  "y", emq(y, y_filtrado), pearson(y, y_filtrado)),
]
for modelo, saida, e, r in linhas_tabela:
    print(f"{modelo:<38}{saida:<7}{e:>10.5f}{np.sqrt(e):>9.4f}{r:>11.4f}{r**2:>8.4f}")

print(f"\nNúmero de parâmetros: RBF dinâmica = {len(w_rbf_x)} pesos por saída (+ σ e 26 centros) | MMQ = 4 por saída")
