# Correct modulus-bound interpretation

This version aligns the code with the protocol-design and implementation chapters.

- The plaintext modulus `p` is checked in the SIMD slot domain.
- The concrete registered query uses `B_M_slot_public * ||y||_1`.
- The dense `d * B_M_slot_public * B_y_slot` value is printed only as a comparison and is explicitly marked unsupported when it exceeds `p`.
- Plaintext polynomials are lifted from `R_p` to `R_q` using centered coefficients. Therefore `B_M_coeff`, `B_y_coeff`, and `B_IP_coeff` are genuine centered coefficient norms and cannot exceed `floor(p/2)`.
- The recursive `N||a||_inf||b||_inf` tracker is a generic diagnostic. If it exceeds `q`, the program says that the selected `q` is not formally certified by that loose inequality.
- The exact execution audit checks the actual simulated representative `v=m+p*eta` and aborts if it reaches `q/2`.

The symbolic modulus formulas in the protocol design remain valid. The correction is to use centered plaintext lifts consistently and to avoid claiming that the concrete `q=2^61-1` satisfies the very loose generic diagnostic when it does not.


## Final raw-representative correction

The symbolic ciphertext-modulus report now uses one consistent decomposition.
For the query circuit it pairs the raw plaintext envelope propagated through
ciphertext multiplication and rotate-and-add with the noise envelope propagated
through the same operations. The canonical encoded scalar is reserved for the
exact post-execution audit. This avoids omitting the polynomial `kappa` in
`F(M*y) = Encode(<M,y>) + p*kappa`.


## Ciphertext modulus correction

The earlier package used `q = 2**61 - 1`, although the implemented
raw-representative sufficient bound was approximately `1.17e20`. This
package sets

```text
q = 295147905179352836353
```

This is a 69-bit prime with `q ≡ 1 (mod 128)` and is above the reported
conservative bound for the configured `N=64` encrypted-query circuit.
Because the value exceeds the signed 64-bit range, `ring.py` now stores ring
coefficients as Python arbitrary-precision integers. The protocol aborts if
a runtime symbolic bound ever exceeds the selected modulus.
