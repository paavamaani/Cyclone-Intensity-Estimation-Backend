import cv2
import copy

def extractKinematics(dist, intensity, spatial_resolution = 4):
    dist_kms = dist * spatial_resolution
    time = (dist_kms / intensity) * 60

    time = round(time, 0)

    hrs = str(int(time // 60)).zfill(2)
    mins = str(int(time % 60)).zfill(2)

    return f'{hrs}:{mins}'

def detectCoastalLine(image, coor):
    coast_lines = {
        "Kutch and Kathiawar": [(275,262), (276,263), (278,267), (281,273), (283,276), (287,277), (290,277), (295,276), (297,274), (296,276), (296,278)],
        "Konkan Coast": [(299,281), (299,283), (298,287), (299,290), (299,292), (299,296), (299,297), (300,299), (300,302), (300,304), (303,306), (303,309), (305,310), (305,312), (306,314), (306,319), (307,320), (307,321), (310,324), (311,326)],
        "Malabar Coast": [(310,326), (311,328), (312,331), (313,333), (314,335), (314,337), (317,339), (317,341), (318,344), (319,347), (320,349), (322,351)],
        "cape comorin": [(326,355)],
        "Coromandel Coast": [(325,355), (327,353), (329,349), (333,349), (334,346), (334,343), (336,342), (339,341), (339,340), (339,337), (339,335), (337,334), (339,333), (339,332), (339,329), (339,326), (340,325)],
        "Northern Circar": [(342,320), (341,315), (341,313), (341,309), (344,309), (347,307), (349,305), (353,304), (353,302), (354,300), (354,299), (356,299), (359,298), (363,295), (363,292), (364,292)],
        "Utkal Plain": [(370,288), (371,287), (373,285), (379,282), (379,280), (381,277), (381,276), (384,274), (386,273)],
        "Bengal Coast": [(384,271), (386,271), (390,271), (392,271), (395,271)]
    }
    
    center = [int(coor[0] + coor[2])//2, int(coor[1] + coor[3])//2]

    critical_landfall = []
    critical_coast_line = ""
    dist = None

    for coast_data in coast_lines.items():
        coast_line = coast_data[0]
        for (x, y) in coast_data[1]:
            temp = ((center[0] - x)**2 + (center[1] - y)**2)**(0.5)
            if dist is None:
                dist = temp
                critical_landfall = [x, y]
                critical_coast_line = coast_line
            elif temp < dist:
                dist = temp
                critical_landfall = [x, y]
                critical_coast_line = coast_line

    return [critical_landfall, critical_coast_line, dist]

def getBestFit(image, results):
    vertical_half = 330
    ind_fit = [[175, 250], [375, 450]]
    crop_coor = copy.deepcopy(ind_fit)

    center = [int(results[0] + results[2])//2, int(results[1] + results[3])//2]

    if center[1] < vertical_half:
        crop_coor[0][1] = int(min(results[0] - 25, ind_fit[0][1]))
    else:
        crop_coor[1][1] = int(max(results[2] + 25, ind_fit[1][1]))

    crop_coor[1][0] = int(max(results[3] + 25, ind_fit[1][0]))

    image = image[crop_coor[0][0]:crop_coor[1][0], crop_coor[0][1]:crop_coor[1][1]]
    
    return image

def drawCoastLine(image, coor, landfall, coast_line):
    coast_lines = {
        "Kutch and Kathiawar": [(275,262), (276,263), (278,267), (281,273), (283,276), (287,277), (290,277), (295,276), (297,274), (296,276), (296,278)],
        "Konkan Coast": [(299,281), (299,283), (298,287), (299,290), (299,292), (299,296), (299,297), (300,299), (300,302), (300,304), (303,306), (303,309), (305,310), (305,312), (306,314), (306,319), (307,320), (307,321), (310,324), (311,326)],
        "Malabar Coast": [(310,326), (311,328), (312,331), (313,333), (314,335), (314,337), (317,339), (317,341), (318,344), (319,347), (320,349), (322,351)],
        "cape comorin": [(326,355)],
        "Coromandel Coast": [(325,355), (327,353), (329,349), (333,349), (334,346), (334,343), (336,342), (339,341), (339,340), (339,337), (339,335), (337,334), (339,333), (339,332), (339,329), (339,326), (340,325)],
        "Northern Circar": [(342,320), (341,315), (341,313), (341,309), (344,309), (347,307), (349,305), (353,304), (353,302), (354,300), (354,299), (356,299), (359,298), (363,295), (363,292), (364,292)],
        "Utkal Plain": [(370,288), (371,287), (373,285), (379,282), (379,280), (381,277), (381,276), (384,274), (386,273)],
        "Bengal Coast": [(384,271), (386,271), (390,271), (392,271), (395,271)]
    }

    center = [int(coor[0] + coor[2])//2, int(coor[1] + coor[3])//2]

    image = cv2.circle(image, (center[0], center[1]), radius = 2, color = (0, 0, 255), thickness = -1)

    for mark in coast_lines[coast_line]:
        image = cv2.circle(image, (mark[0], mark[1]), radius = 2, color = (0, 0, 255), thickness = -1)
    image = cv2.line(image, center, landfall, color = (0, 0, 255), thickness = 1)

    image = getBestFit(image, coor)
    return image