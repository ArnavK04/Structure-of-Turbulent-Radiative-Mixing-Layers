ffmpeg -framerate 5 -i ./bin/KH_tempPDF_snapshot_%05dC1.png -c:v libx264 -pix_fmt yuv420p PDFs.mp4
ffmpeg -framerate 5 -i ./bin/KH_2D_snapshot_Z_0%05dC1.png -c:v libx264 -pix_fmt yuv420p Z_movies_0.mp4
ffmpeg -framerate 5 -i ./bin/KH_2D_Z_0_%05d_C1grey.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p 4paper_plots.mp4
ffmpeg -framerate 5 -i ./bin/KH_2D_snapshot_X_0%05dC1.png -c:v libx264 -pix_fmt yuv420p X_movies_0.mp4
ffmpeg -framerate 5 -i ./bin/KH_2D_snapshot_Y_0%05dC1.png -c:v libx264 -pix_fmt yuv420p Y_movies_0.mp4
ffmpeg -framerate 5 -i ./bin/KH_1Dhz_snapshot_%05dC1.png -c:v libx264 -pix_fmt yuv420p 1D_movies.mp4
ffmpeg -framerate 5 -i ./bin/KH_mean_fluxes_%05dC1.png -c:v libx264 -pix_fmt yuv420p mean_fluxes.mp4
ffmpeg -framerate 5 -i ./bin/KH_tempPDF_slice_x_%d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p sliced_pdfs_x.mp4
ffmpeg -framerate 5 -i ./bin/KH_tempPDF_slice_y_%d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p sliced_pdfs_y.mp4
ffmpeg -framerate 5 -i ./bin/KH_tempPDF_slice_z_%d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p sliced_pdfs_z.mp4
mkdir snaps
cp *mp4 snaps
cp ./bin/header.txt snaps
cp ./bin/*00000C1.npz snaps
cp ./bin/*00000_C1.npz snaps
cp ./bin/*00125C1.npz snaps
cp ./bin/*0085C1.npz snaps
cp ./bin/*00125_C1.npz snaps
cp ./bin/*0085_C1.npz snaps
cp KH.hydro.hst snaps
cp ./bin/KH_1D_arrays_snapshot0_125_00085_C1_y_lims_corrected.npz snaps
cp ./bin/KH_spacetime_trml_frame_time_averaged0to125with1.png snaps
cp ./bin/KH_spacetime_sim_frame_time_averaged0to125with1.png snaps
cp ./bin/*time_averaged0to125with1.npz snaps
cp ./bin/*time_averaged36to125with1.npz snaps
cp ./bin/*time_averaged36to65with1.npz snaps
cp ./bin/*time_averaged66to95with1.npz snaps
cp ./bin/*time_averaged96to125with1.npz snaps
