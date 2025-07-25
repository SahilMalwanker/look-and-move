
clear all,    close all; clc

% 1. Forward Kinematics
syms q1 q2 q3 q4 q5 q6 real

% a) 
DH = [ q1,     0,   0,       -pi/2;
       q2,     0,       0.237,    0;
       q3,     0,       0,   pi/2;
       q4,     0.297,   0,        pi/2;
       q5,     0,       0,       -pi/2;
       q6,     0.126,   0,        0];

% b)
function T = DH_transform(theta, d, a, alpha)
    T = [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta);
         sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
              0    ,         sin(alpha)    ,         cos(alpha)    ,     d;
              0    ,             0         ,             0         ,     1];
end

T = sym(eye(4));
for i = 1:6
    theta = DH(i,1);
    d = DH(i,2);
    a = DH(i,3);
    alpha = DH(i,4);
    T = T * DH_transform(theta, d, a, alpha);
end
T06 = T;

% Add tool offset as before
T_tool = eye(4);
T_tool(3,4) = 0.06;
T_tcp = simplify(T06 * T_tool);

% c)
px = simplify(T_tcp(1,4));
py = simplify(T_tcp(2,4));
pz = simplify(T_tcp(3,4));
R = T_tcp(1:3, 1:3);

% Extract Euler angles manually from symbolic R
gamma = atan2(R(2,1), R(1,1));       % Yaw
beta  = atan2(-R(3,1), sqrt(R(3,2)^2 + R(3,3)^2));  % Pitch
alpha = atan2(R(3,2), R(3,3));       % Roll

X0 = [px; py; pz; alpha; beta; gamma];


% d) Test case evaluation
testcases_deg = [8.88, -127.32, 171.91, 167.63, 46.09, 8.65;
                -45.73, -107.44, 161.60, -89.57, 90.59, -54.16;
                50.06, -104.34, 159.33, 87.09, 94.15, 54.89;
                37.53, -104.47, 183.48, 50.94, 91.77, 125.39;
                -33.41, -107.24, 185.63, -51.75, 88.23, -127.44];

testcases_rad = deg2rad(testcases_deg);  

for i = 1:size(testcases_rad,1)
    Xi   = double(subs(X0, [q1,q2,q3,q4,q5,q6], testcases_rad(i,:)));
    ang  = rad2deg(Xi(4:6));        % ⟵ convert to degrees
    fprintf(['Test case %d:\n' ...
             ' [px,py,pz] = [%.4f, %.4f, %.4f] m\n' ...
             ' [A,B,C]   = [%.2f°, %.2f°, %.2f°]\n\n'], ...
             i, Xi(1), Xi(2), Xi(3), ang);
end



% All five test cases position the robot's tool at the same height (around 400-500mm) with 
% identical downward orientation. Only the X and Y coordinates change - like moving a pen 
% vertically down to different spots on a table. This setup is perfect for 2D visual tracking 
% tasks where we just need to follow horizontal movements while maintaining steady downward pressure. 
% The joint angles vary but work together to keep the tool stable, showing the robot can reliably 
% reach different points in its workspace while maintaining consistent positioning - exactly what we 
% need for precise visual servoing.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2. Visual Control

% a)
% To increase framerate and reduce computational load.

% b)
full_width = 4056;
used_width = 640;

dPx = 1.55e-6; % [m] (original pixel size)
dPx_red = dPx * (full_width / used_width);  % in meters

fprintf("Reduced pixel size: %.2f µm\n", dPx_red * 1e6);

% c)
% no,  it’s not required to reach the desired image feature, 
% since control relies only on 2D image displacement.

% d)
syms cx cy cz l real

l = 6e-3;
sx = cx * l / cz;
sy = cy * l / cz;

Jv = jacobian([sx; sy], [cx, cy, cz]); 

disp('Image Jacobian Jv:');
disp(simplify(Jv));
