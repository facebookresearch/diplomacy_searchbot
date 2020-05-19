import diplomacy
import time

N = 1000

if __name__ == "__main__":

    all_orders = {
        "AUSTRIA": ["A BUD - SER", "F TRI - ALB", "A VIE - GAL"],
        "ENGLAND": ["A LVP - EDI", "F EDI - NWG", "F LON - NTH"],
        "FRANCE": ["F BRE - MAO", "A PAR - PIC", "A MAR - BUR"],
        "GERMANY": ["F KIE - DEN", "A MUN - RUH", "A BER - KIE"],
        "ITALY": ["F NAP - ION", "A ROM - APU"],
        "RUSSIA": ["F STP/SC - BOT", "F SEV - BLA", "A MOS - UKR", "A WAR - GAL"],
        "TURKEY": ["F ANK - BLA", "A CON - BUL", "A SMY - CON"],
    }

    t_elapsed = 0

    for _ in range(N):
        game = diplomacy.Game()
        for power, orders in all_orders.items():
            game.set_orders(power, orders)

        t_start = time.time()
        game.process()
        t_elapsed += time.time() - t_start

    print("{:.3f}s / {} = {:.3f} us / process".format(t_elapsed, N, t_elapsed / N * 1e6))
