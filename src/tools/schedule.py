import pprint

class ScheduleLoader(object):
    def __init__(self, schedule):
        pp = pprint.PrettyPrinter(indent=2)
        print('\n=== schedule ===\n')
        pp.pprint(schedule)
        self.schedule = schedule

    def __repr__(self):
        return pp.pformat(schedule, indent=2)
