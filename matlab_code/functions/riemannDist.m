function dist = riemannDist(A, B)
    % RIEMNNDIST  Affine-invariant Riemannian distance between SPD matrices A,B
    %
    %   dist = riemannDist(A, B)
    %
    %   Computes   || logm(A^(-1/2) * B * A^(-1/2)) ||_Fro.

    % build the congruence-transformed matrix
    M = sqrtm(A) \ B / sqrtm(A);
    % Frobenius norm of its matrix logarithm
    dist = norm(logm(M), 'fro');
end