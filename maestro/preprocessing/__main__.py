import argparse

def main():
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='mode')
    sub_parsers.required = True


    # ========== Augmentation ============
    # Augmentation command
    augmentation_parser = sub_parsers.add_parser('augment')
    
    augmentation_parser.add_argument('data_path', type=str, help='Path to the directory containing the wav files')
    augmentation_parser.add_argument('--sample_rate', type=int, default=48000, help='Sample rate of the output audio files')
    augmentation_parser.add_argument('--length', type=int, default=None, help='Number of augmented files to generate')
    augmentation_parser.add_argument('--aug_factor', type=int, default=None, help='Number of augmented files to generate per input file')
    augmentation_parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use for augmentation')


    # ========== Laion clap preprocessing ============

    # Preprocessing command
    laion_clap_parser = sub_parsers.add_parser('laion_clap')

    laion_clap_parser.add_argument('--target_sample_rate', type=int, default=48000, help='Sample rate of the output audio files')
    laion_clap_parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use for preprocessing')
    laion_clap_parser.add_argument('--resample_only', action='store_true', help='Only resample the audio files')

    laion_clap_subparsers = laion_clap_parser.add_subparsers(dest='data_origin')


    #process waveforms
    laion_process_parser = laion_clap_subparsers.add_parser('process')
    laion_process_parser.add_argument('data_path', type=str, help='Path to the directory containing the wav files')
    laion_process_parser.add_argument('out_path', type=str, help='Path to the directory to save the processed files')
    laion_process_parser.add_argument('--sample_rate', type=int, default=48000, help='Sample rate of the output audio files')
    #synth
    laion_clap_synth_parser = laion_clap_subparsers.add_parser('synth')    

    laion_clap_synth_parser.add_argument('data_path', type=str, help='Path to the directory containing the wav files')
    laion_clap_synth_parser.add_argument('out_path', type=str, help='Path to the directory to save the processed files')

    #gtzan
    laion_clap_gtzan_parser = laion_clap_subparsers.add_parser('gtzan')

    laion_clap_gtzan_parser.add_argument('input_dir', type=str, help='Path to the directory containing the GTZAN dataset')
    laion_clap_gtzan_parser.add_argument('output_dir', type=str, help='Path to the directory to save the processed files')



    args = parser.parse_args()

    if args.mode == 'laion_clap':
        if args.data_origin == 'synth':
            from maestro.preprocessing.laion_clap import preprocess_synth
            preprocess_synth(data_path=args.data_path, out_path=args.out_path, target_sample_rate=args.target_sample_rate, num_workers=args.num_workers, resample_only=args.resample_only)
        elif args.data_origin == 'gtzan':
            from maestro.preprocessing.laion_clap import preprocess_gtzan
            preprocess_gtzan(input_dir=args.input_dir, output_dir=args.output_dir, target_sample_rate=args.target_sample_rate, num_workers=args.num_workers, resample_only=args.resample_only)
        elif args.data_origin == 'process':
            from maestro.preprocessing.laion_clap import process_waveforms
            process_waveforms(input_dir=args.data_path, output_dir=args.out_path, sample_rate=args.sample_rate)
        else:
            raise ValueError(f"Unknown data origin: {args.data_origin}")
    elif args.mode == 'augment':
        from maestro.preprocessing.augmentation import augment_audio
        augment_audio(data_path=args.data_path, sample_rate=args.sample_rate, length=args.length, aug_factor=args.aug_factor)
        #TODO: validate args
    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == '__main__':
    main()