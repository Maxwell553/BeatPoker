"""Consistent policy selection for terminal and browser launchers."""


def add_policy_arguments(parser):
    parser.add_argument('--native-iterations',type=int,default=1000,
                        help='Maximum native search iterations within --seconds (default: 1000)')
    parser.add_argument('--native-uncompressed',dest='native_compression',action='store_false',
                        help='Use float32 native solver storage instead of compressed storage; increases memory use')
    parser.add_argument('--blueprint',help='Optional experimental ECFR preflop checkpoint for native search')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--full-blueprint',dest='full_blueprint',action='store_true',default=None,
                       help='Use the published full policy (default for unraked play)')
    modes.add_argument('--native',dest='full_blueprint',action='store_false',default=None,
                       help='Use chart/preflop policy with native postflop search')
    parser.add_argument('--resolve-from',choices=('turn','river'),
                        help='Refine the full policy with exact-card search from this street')
    parser.add_argument('--river-gadget',action='store_true',
                        help='Experimental river refinement with a per-hand value gate')


def resolve_policy_arguments(args,parser):
    from .bot import SolverConfig
    try:SolverConfig(iterations=args.native_iterations)
    except ValueError as exc:parser.error(str(exc))
    if args.full_blueprint is None:
        args.full_blueprint=not args.blueprint and args.rake=='none'
    if args.full_blueprint and (args.blueprint or args.rake!='none'):
        parser.error('--full-blueprint requires --rake none and cannot combine with --blueprint')
    if args.resolve_from and not args.full_blueprint:
        parser.error('--resolve-from requires the full policy')
    if args.river_gadget and (not args.full_blueprint or args.resolve_from):
        parser.error('--river-gadget requires the full policy and cannot combine with --resolve-from')
    return args
