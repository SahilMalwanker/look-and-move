% Define points
rho_a = [251, 190, 0];
rho_b = [345,  35, 0];
rho_c = [116, 105, 0];

% Step 1: Vector from rho_a to rho_b → this is your x-direction
x_vec = rho_b - rho_a;
x_vec = x_vec / norm(x_vec);  % normalize

% Step 2: Get y-direction vector
% Since all points lie in the XY plane (z=0), we can take cross with z-axis
z_axis = [0, 0, 1];
y_vec = cross(z_axis, x_vec);  % this gives you a vector in the XY plane perpendicular to x_vec
y_vec = y_vec / norm(y_vec);   % normalize

% Step 3: Vector from rho_a to rho_c
u = rho_c - rho_a;

% Step 4: Project u onto new basis vectors
x0img = dot(u, x_vec);
y0img = dot(u, y_vec);

% Display results
disp(['x0img = ', num2str(x0img)]);
disp(['y0img = ', num2str(y0img)]);
