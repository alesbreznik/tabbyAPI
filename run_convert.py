import sys
from exllamav3.conversion.convert_model import parser, prepare, main

if __name__ == "__main__":
    args = parser.parse_args()
    in_args, job_state, ok, msg = prepare(args)
    if not ok:
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)
    main(in_args, job_state)
