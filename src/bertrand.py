import numpy as np
import sympy as sp


class Bertrand:
    """Bayesian Bertrand competition with Beta(1,l) prior and power-demand kernel.

    Kernel:  k^s(p, c) = (1-p)^s * (p - c)

    Strategies are piecewise-linear functions on [0,1] parameterised by their
    per-piece slopes x = (x_1, ..., x_m).  The game gradient v_i(x) is
    computed in closed form via the Proposition integrals (SymPy).

    Parameters
    ----------
    s : int >= 0
        Demand exponent.  s=0 -> all-or-nothing, s=1 -> linear, s=2 -> quadratic.
    l : int >= 1
        Beta(1, l) prior parameter (l=1 -> uniform).  Must be a positive integer
        so that the weight l*(1-c)^{ln-2} is polynomial in c.
    n : int >= 2
        Number of players.
    m : int >= 1
        Number of pieces (strategy dimension).

    Attributes
    ----------
    bne : np.ndarray, shape (m,)
        Symmetric BNE slope vector.  Each entry equals
        x* = (l(n-1) + s) / (l(n-1) + s + 1).

    Methods
    -------
    gradient(x) : np.ndarray
        Game gradient at x, shape (m,) or (N, m).
    get_gradient_sym() : (tuple of Symbol, list of Expr)
        Closed-form symbolic gradient via the Proposition integrals.
    """

    def __init__(self, s: int, l: int, n: int, m: int):
        assert isinstance(s, int) and s >= 0, "s must be a non-negative integer"
        assert isinstance(l, int) and l >= 1, "l must be a positive integer"
        assert isinstance(n, int) and n >= 2, "n must be an integer >= 2"
        assert isinstance(m, int) and m >= 1, "m must be a positive integer"
        # l*n >= 2 is guaranteed since l >= 1 and n >= 2

        self.s = s
        self.l = l
        self.n = n
        self.m = m

        # Symmetric BNE slope  x* = (l(n-1)+s) / (l(n-1)+s+1)
        alpha = l * (n - 1) + s
        self.bne = (alpha / (alpha + 1)) * np.ones(m)

        # Pre-compile the gradient once at construction time
        x_sym, grad_sym = self.get_gradient_sym()
        self._grad_num = sp.lambdify([x_sym], grad_sym, "numpy")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Game gradient at x.

        Parameters
        ----------
        x : np.ndarray, shape (m,) or (N, m)
            Slopes of the piecewise-linear strategy.

        Returns
        -------
        np.ndarray
            Gradient at x, same shape as input.
        """
        return np.array(self._grad_num(x.T)).T

    # ------------------------------------------------------------------
    # Symbolic gradient (closed form via Proposition)
    # ------------------------------------------------------------------

    def get_gradient_sym(self):
        """Return (x_symbols, grad_list) for the closed-form gradient.

        Implements Proposition: v_i^{(s,l,n)}(x) via SymPy integration.

        Returns
        -------
        x_sym : tuple of Symbol
            SymPy symbols (x1, ..., x_m).
        grad  : list of Expr
            grad[i] = v_{i+1}(x), the (i+1)-th gradient component.
        """
        m, s, l, n = self.m, self.s, self.l, self.n
        x_sym = sp.symbols(f"x1:{m + 1}")          # x1, x2, ..., x_m
        c, p  = sp.symbols("c p", real=True)

        # Kernel  k^s(p,c) = (1-p)^s * (p-c)  and  ∂_p k^s
        k      = (1 - p) ** s * (p - c)
        dk_dp  = sp.diff(k, p)

        # Prior weight  l*(1-c)^{ln-2}
        def weight(c_var):
            return l * (1 - c_var) ** (sp.Rational(l * n) - 2)

        grad = []
        for i in range(1, m + 1):
            v_i = sp.Integer(0)

            # ── Sum terms j = 1, …, i-1 ──────────────────────────────────────
            for j in range(1, i):
                x_j     = x_sym[j - 1]
                x_ge_j  = sum(x_sym[k_ - 1] for k_ in range(j, m + 1))
                beta_j  = x_j * (c - sp.Rational(j - 1, m)) + 1 - x_ge_j / m

                integrand_j = (
                    weight(c)
                    * (
                        l * (n - 1) * k.subs(p, beta_j) / x_j
                        - (1 - c) * dk_dp.subs(p, beta_j)
                    )
                )
                v_i += sp.Rational(1, m) * sp.integrate(
                    integrand_j,
                    (c, sp.Rational(j - 1, m), sp.Rational(j, m)),
                )

            # ── i-th (diagonal) term ─────────────────────────────────────────
            x_i    = x_sym[i - 1]
            x_ge_i = sum(x_sym[k_ - 1] for k_ in range(i, m + 1))
            beta_i = x_i * (c - sp.Rational(i - 1, m)) + 1 - x_ge_i / m

            integrand_i = (
                weight(c)
                * (
                    l * (n - 1) * k.subs(p, beta_i) / x_i
                    - (1 - c) * dk_dp.subs(p, beta_i)
                )
            )
            v_i += sp.integrate(
                (sp.Rational(i, m) - c) * integrand_i,
                (c, sp.Rational(i - 1, m), sp.Rational(i, m)),
            )

            grad.append(sp.simplify(v_i))

        return x_sym, grad
