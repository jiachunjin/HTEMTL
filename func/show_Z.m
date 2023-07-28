function [h] = show_Z(Z)
%SHOW_Z Summary of this function goes here
%   Detailed explanation goes here

figure
Z = 0.5 *(abs(Z) + abs(Z'));
Z = Z / 10;
Z = Z / (max(max(Z)) - min(min(Z)));
colormap(flipud(gray));
h = imagesc(Z);
colorbar;

end

