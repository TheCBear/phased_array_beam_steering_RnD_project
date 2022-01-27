%% Antenna Properties 
% Design antenna at frequency 8000000000Hz
frequency = 8e9;
antennaObject = design(patchMicrostrip,8000000000);
% Properties changed 
antennaObject.Length = 0.012;
antennaObject.Width = 0.009;
antennaObject.Height = 0.00155;
antennaObject.GroundPlaneLength = 0.0187;
antennaObject.GroundPlaneWidth = 0.0187;
antennaObject.FeedOffset = [0.003 0];
% Update substrate properties 
antennaObject.Substrate.Name = 'Air';
antennaObject.Substrate.EpsilonR = 3.48;
antennaObject.Substrate.LossTangent = 0.0037;
antennaObject.Substrate.Thickness = 0.00155;

%% Visualize
theta = -90:5:90;
phi = -180:5:180;
pattern(antennaObject, 8e9, phi, theta)

%% Simulate
% Simulation settings
theta = -90:0.5:90;
phi = -180:0.5:180;
%theta = -90:5:90;
%phi = -180:5:180;
tic
[pat, azimuth, elevation] = pattern(antennaObject, 8e9, phi, theta);
toc
%% Save
gain = pat;
pattern_type = "gain";
save('ant_gain_pat.mat', 'gain', 'azimuth', 'elevation', 'frequency',...
        'pattern_type');