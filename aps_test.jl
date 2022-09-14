using GaussianRandomFields
using Plots

cov = CovarianceFunction(2, Matern(1/4, 3/4))

# Vertical
ny = 50;
pts_y = range(0, stop=10, length=ny)

# Horizontal
nx = 60;
pts_x = range(0, stop=4, length=nx)

grf = GaussianRandomField(cov, CirculantEmbedding(), pts_y, pts_x)

z = sample(grf);
#heatmap(z)

f = (z .< 0);
heatmap(f)

# Make a well and superimpose on image
well = [zeros(20); ones(10); zeros(20)]
i_well = 20
x_well = pts_x[i_well]
f[:, i_well] = well;
heatmap(f)

# Adapt threshold to well
threshold_well = -3 * ones(ny);
threshold_well[20:30] .= 3;

# krige to spread values to well vicinity
# Get covariance values
#sig_11 = apply(cov, pts_x, pts_y)
sig_22 = apply(cov, [x_well], pts_y);
sig_12 = zeros(nx*ny, ny);
for i = 1:nx
    for j = 1:ny
        for k = 1:ny
            distance = sqrt((pts_x[i] - x_well)^2 + (pts_y[j] - pts_y[k])^2);
            sig_12[ny * (i - 1) + j, k] = apply(cov, [0.0 distance], [0.0 0.0])[1, 2];
        end
    end
end

heatmap(reshape(sig_12[:, 30], ny, nx))

threshold_adapted = reshape(sig_12 * (sig_22 \ threshold_well), ny, nx);

heatmap(threshold_adapted)

f_tally = zeros(ny, nx);
n_sample = 100

for b = 1:n_sample
    # Draw grf realization
    z_new = sample(grf);
    # Truncate realization
    f_new = (z_new .< threshold_adapted);
    f_tally += f_new;
end

heatmap(f_tally / n_sample)


