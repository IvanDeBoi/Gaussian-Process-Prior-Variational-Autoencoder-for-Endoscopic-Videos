%% Build ocam_model from txt and undistort

%------------- 1) Build ocam_model (pol + invpol) -------------%
ocam = load_ocam_from_txt('../Data/C3VDv2 CB/camera_intrinsics.txt');

% choose range of theta you care about (0..~1.2 rad ~ 70 deg)
theta_max = 1.2;
n_samples = 2000;

theta = linspace(0, theta_max, n_samples);        % angle
pol   = ocam.pol(:)';                             % [a0..a4]

r = zeros(size(theta));
for i = 1:numel(pol)
    r = r + pol(i)*theta.^(i-1);                  % r(theta)
end

% build inverse mapping theta(r) by sampling + polyfit
r_min = min(r); r_max = max(r);
r_grid = linspace(r_min, r_max, n_samples);
theta_interp = interp1(r, theta, r_grid, 'linear', 'extrap');

deg_inv = 9;                                      % degree of invpol
inv_coeff = polyfit(r_grid, theta_interp, deg_inv); % highest power first

% ocam.invpol is lowest power first, so flip
ocam.invpol        = fliplr(inv_coeff);
ocam.length_pol    = numel(ocam.pol);
ocam.length_invpol = numel(ocam.invpol);

%------------- 2) Read omni image -----------------------------%
I  = imread('../Data/C3VDv2 CB/C3VD_0030.png');
Id = im2double(I);
[H,W,C] = size(Id);

%------------- 3) Define output pinhole view ------------------%
outWidth  = 800;
outHeight = 600;
fOut      = 600;
cxOut     = outWidth/2;
cyOut     = outHeight/2;

[xu, yu] = meshgrid(1:outWidth, 1:outHeight);
xNorm = (xu - cxOut)/fOut;
yNorm = (yu - cyOut)/fOut;

Z = ones(size(xNorm));
normP = sqrt(xNorm.^2 + yNorm.^2 + Z.^2);
X = xNorm ./ normP;
Y = yNorm ./ normP;
Z = Z      ./ normP;

%------------- 4) Use world2cam with invpol -------------------%
[ud, vd] = world2cam_from_invpol(X, Y, Z, ocam);

ud = min(max(ud, 1), W);
vd = min(max(vd, 1), H);

x = 1:W; y = 1:H;
out = zeros(outHeight, outWidth, C);
for ch = 1:C
    out(:,:,ch) = interp2(x, y, Id(:,:,ch), ud, vd, 'linear', 0);
end

%------------- 5) Show result ---------------------------------%
figure;
subplot(1,2,1); imshow(I);   title('Original (omnidirectional)');
subplot(1,2,2); imshow(out); title('Undistorted (pinhole-like)');

%% ================= helper functions =========================
function ocam = load_ocam_from_txt(filename)
    txt = fileread(filename);
    ocam.width  = parse_val(txt,'width');
    ocam.height = parse_val(txt,'height');
    ocam.cx     = parse_val(txt,'cx');
    ocam.cy     = parse_val(txt,'cy');
    ocam.c      = parse_val(txt,'c');
    ocam.d      = parse_val(txt,'d');
    ocam.e      = parse_val(txt,'e');
    ocam.pol    = [ ...
        parse_val(txt,'a0'), ...
        parse_val(txt,'a1'), ...
        parse_val(txt,'a2'), ...
        parse_val(txt,'a3'), ...
        parse_val(txt,'a4')];
end

function v = parse_val(txt, key)
    expr   = [key '\s*=\s*([^\r\n]+)'];
    tokens = regexp(txt, expr, 'tokens', 'once');
    if isempty(tokens)
        error('Key "%s" not found', key);
    end
    v = str2double(strtrim(tokens{1}));
end

function [u,v] = world2cam_from_invpol(X, Y, Z, ocam)
    invpol = ocam.invpol(:)';                 % [b0..bn], theta(r)
    rho = sqrt(X.^2 + Y.^2);
    rho(rho==0) = eps;

    theta = acos(Z);

    % estimate image radius r from theta using inverse poly approx:
    r = zeros(size(theta));
    for i = 1:numel(invpol)
        r = r + invpol(i)*theta.^(i-1);
    end

    x = X ./ rho;
    y = Y ./ rho;

    u = x.*r*ocam.c + y.*r*ocam.d + ocam.cx;
    v = x.*r*ocam.e + y.*r       + ocam.cy;
end
