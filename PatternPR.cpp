//***** CANON MEDICAL SYSTEMS CORPORATION ** COMPANY CONFIDENTIAL *****
///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2007-2013 Canon Medical Systems Corporation, All Rights Reserved
//  Creted: 13-03-21 wheaton
///////////////////////////////////////////////////////////////////////////////
#include <MR/common/stlpdefs.h>
#include <iomanip>
using std::endl;

#include <MR/libseqobj/SeqObjErrors.h>
#include <MR/libpattern/MomentTable.h>
#include <MR/libseqobj/Loop.h>
#include <MR/common/vf_appcodes.h>
#include <MR/common/tmath.h>
#include <MR/libSeqLink/TuneLinkScope.h>
#include <MR/libpattern/PatternPR.h>
#include <MR/libseqobj/Dimension.h>

using namespace MRMPlus;

static const int32_t s_gold_numerator     = 28657;
static const int32_t s_gold_denomominator = 46368;

void AngleMgr::saveToStudyDB(SeqDBSave& par)
{
    if (m_active)
    {
        // save cycle and numSample
        par.trajectoryCoord.resize(4);
        par.trajectoryCoord[0] = (flt32_t)m_cycle;
        par.trajectoryCoord[1] = (flt32_t)m_sample;
        par.trajectoryCoord[2] = (flt32_t)m_cycle_tmplt;
        par.trajectoryCoord[3] = (flt32_t)m_sample_tmplt;
    }
}

void PatternPR::EpvInit()
{
    PatternBase::init();

    //m_anglePattern = PR_RECURSIVE;
    m_anglePattern = PR_180;
    m_cycle.clear();
    m_numSample.clear();
    // For PR_RECURSIVE and PR_GEN_GOLDEN
    m_numSpoke = 3;
    m_numTemplate.clear();

    m_angleMgr.init();
}

void PatternPR::setParams(const SeqParam& sp)
{
    PatternBase::setParams(sp);

    string base = "radial";
    if (sp.setMap(base))
    {
        int ival(0);
        if (sp.getValue(VRT_SL, "anglePattern", &ival))
        {
            // PR_FREE_ANGLE = 0, PR_180 = 1, PR_360 = 2, PR_RECURSIVE = 8
            m_anglePattern = RadialAnglePattern(ival);
        }
        sp.getValue(VRT_FD, "cycle", &m_cycle);
        sp.getValue(VRT_SL, "numSample", &m_numSample);
        sp.getValue(VRT_SL, "numSpoke", &m_numSpoke);
        sp.getValue(VRT_SL, "numTemplate", &m_numTemplate);
    }

    if (m_anglePattern == PR_180)
        m_cycle = 0.5;
    else if (m_anglePattern == PR_360 || m_anglePattern == PR_RECURSIVE || m_anglePattern == PR_GEN_GOLDEN)
        m_cycle = 1.0;
}

void PatternPR::makeIndexTable(MPlusInt32Vec &indexTable)
{
    if (m_anglePattern == PR_RECURSIVE)
    {
        SeqRoot *seqRoot = 0;
        int check = 1;
        int numSpoke = m_numSpoke.getValue();
        int numAcq = static_cast<int32_t>(indexTable.size());
        LoopList loops;
        while (check < numAcq)
        {
            Loop *loop = new Loop(seqRoot, "Rec");
            loops.push_back(loop);
            loop->setRange(0, check * (numSpoke - 1), check);
            check *= numSpoke;
        }
        indexTable.clear();
        indexTable.reserve(check);
        LoopSet recLps = loops;
        do
        {
            int index = 0;
            for (size_t i = 0; i < recLps.size(); ++i)
            {
                index += recLps[i].getIndex();
            }
            indexTable.push_back(index);
        }
        while (recLps.next());

        for (size_t i = 0; i < loops.size(); ++i)
        {
            delete loops[i];
        }
        m_numSample = check;
    }
    else if (m_anglePattern == PR_GEN_GOLDEN)
    {
        int numSpoke = m_numSpoke.getValue();
        if (numSpoke > 0)
        {
            for (size_t i = 0, sz = indexTable.size(); i < sz; ++i)
            {
                size_t outerIndex = i / numSpoke;
                size_t innerIndex = i % numSpoke;
                size_t index      = ((s_gold_denomominator - s_gold_numerator) * outerIndex) % s_gold_denomominator
                                    + s_gold_denomominator * innerIndex;
                indexTable[i] = static_cast<int32_t>(index);
            }
        }
        else
        {
            TMR_ASSERT_LOG(false, "numSpoke must be positive integer.");
        }
    }
    else
    {
        for (size_t i = 0, sz = indexTable.size(); i < sz; ++i)
        {
            indexTable[i] = static_cast<int32_t>(i);
        }
    }
}

void PatternPR::makeRotationTable()
{
    CMRTrace trace(TSEQENCODE_TRACE_LETTER);
    bool debug = trace.CheckTraceLevel();
    if (debug)
        trace << __FUNCTION__ << std::endl;

    LoopSet lps = m_dim->getLoopList();
    int32_t numAcq = lps.encodeDim();

    if (!m_numSample.isActive())
        m_numSample = numAcq;

    MPlusInt32Vec indexTable(numAcq);
    makeIndexTable(indexTable);

    m_angleMgr.activate();

    const flt64_t golden_ratio = (sqrt(5.0) - 1) / 2.0;
    flt64_t angle_base;
    switch (m_anglePattern)
    {
    case PR_180:
    case PR_360:
    case PR_RECURSIVE:
    case PR_FREE_ANGLE:
        break;

    case PR_GOLDEN_180:
        angle_base = 0.5 * golden_ratio;
        m_cycle = angle_base * m_numSample;
        break;

    case PR_GOLDEN_360:
        angle_base = 1.0 * golden_ratio;
        m_cycle = angle_base * m_numSample;
        break;

    case PR_GEN_GOLDEN:
        m_numSample = m_numSpoke * s_gold_denomominator;
        if (m_numSample >= (1 << m_trajectorysort.getIndexBit()))
        {
            throw MRException("PatternPR::makeRotationTable", "invalid numSpokes", SEQOBJ_SG_DESIGNER_QUIT);
        }
        break;

    default:
        throw MRException("PatternPR::makeRotationTable", "invalid angle pattern", SEQOBJ_SG_DESIGNER_QUIT);
        break;
    }
    m_angleMgr.setCycle(m_cycle, m_numSample, 1, m_numTemplate);

    int32_t prjidx = 0;
    int32_t tmplt = 0;
    int32_t counter = 0;
    flt64_t RadialAngle = 0;

    int table = 0;
    lps.reset();
    do
    {
        if (lps.isCond())
        {
            // conditioning
            prjidx = 0;
            (*m_EncodeTable)[table].acqIndex = ACQINDEX_NOACQ;
            (*m_EncodeTable)[table].attrib = VFC_SRTTBL_ATTRB_CONDITION;
            if (debug)
                trace << "conditioning:" << lps << std::endl;
        }
        else if (lps.isTempl())
        {
            prjidx = tmplt++;
            RadialAngle = m_angleMgr.getAngle(prjidx, true);
            (*m_EncodeTable)[table].acqIndex = lps.reconOffset();
            (*m_EncodeTable)[table].attrib = VFC_SRTTBL_ATTRB_TEMPLATE;
            if (debug)
                trace << "template:" << lps << " --- " << "Angle:" << (180. * RadialAngle / M_PI) << std::endl;
        }
        else
        {
            prjidx = indexTable[counter++];
            RadialAngle = m_angleMgr.getAngle(prjidx);
            (*m_EncodeTable)[table].acqIndex = lps.reconOffset();
            if (debug)
                trace << "lps:" << lps << " angle " << (180. * RadialAngle / M_PI) << std::endl;
        }
        (*m_EncodeTable)[table].kIndex = prjidx;
        (*m_EncodeTable)[table].freqEncode[ax_x] = cos(RadialAngle);
        (*m_EncodeTable)[table].freqEncode[ax_y] = sin(RadialAngle);
        (*m_EncodeTable)[table].freqEncode[ax_z] = 0;
        ++table;
    }
    while (lps.next());
}

void PatternPR3D::makeRotationTable()
{
    int32_t prjidx;
    int32_t tmpIndx;
    flt64_t zgrdmod;

    LoopSet lps = m_dim->getLoopList();
    LoopSet prjLps;
    // move outermost loop to innermost
    for (size_t i = 1, sz = lps.size(); i < sz; ++i)
    {
        prjLps.push_back(lps[i]);
    }
    prjLps.push_back(lps[0]);
    int32_t numAcq = lps.reconDim();

    lps.reset();
    do
    {
        EncodeData &encodeData = (*m_EncodeTable)[lps];
        if (lps.isCond())
        {
            // conditioning
            prjidx = 0;
            tmpIndx = 0;
            zgrdmod = -1;
            encodeData.acqIndex = ACQINDEX_NOACQ;
            encodeData.attrib = VFC_SRTTBL_ATTRB_CONDITION;
        }
        else
        {
            if (!m_params.m_newReconFlag)
            {
                prjidx = prjLps.reconOffset();
                tmpIndx = lps.reconOffset();
            }
            else
            {
                if (m_params.m_encodeType == VFC_ENCODE_SEQUENTIAL)
                {
                    prjidx = lps.reconOffset();
                }
                else
                {
                    prjidx = prjLps.reconOffset();
                }
                tmpIndx = prjidx;
            }

            zgrdmod = (2.0 * prjidx - numAcq + 1) / (flt64_t)numAcq;
            encodeData.acqIndex = lps.reconOffset();
        }
        encodeData.kIndex = tmpIndx;
        encodeData.freqEncode[ax_x] = cos(sqrt(M_PI * numAcq) * asin(zgrdmod)) * sqrt(1.0 - zgrdmod * zgrdmod);
        encodeData.freqEncode[ax_y] = sin(sqrt(M_PI * numAcq) * asin(zgrdmod)) * sqrt(1.0 - zgrdmod * zgrdmod);
        encodeData.freqEncode[ax_z] = zgrdmod;
    }
    while (lps.next());
}

void PatternPETRA::makeCartesianTable(int32_t numMissingPts)
{
    // Calculate Cartesian section
    int cartCubeDim = numMissingPts * 2 + 1;
    flt64_t petraStep = 1.0 / (flt64_t)(numMissingPts + 1);

    m_CartesianEncodeData.resize(cartCubeDim * cartCubeDim * cartCubeDim, EncodeDataMixed());

    int32_t idx(0), outerIdx(m_params.numCartEncodes), kidx(0);
    F64Vec3 k;
    flt64_t dist;
    int oddx = 1;
    int oddy = 1;
    int xval, yval;
    for (int zz = 0; zz < cartCubeDim; zz++)
    {
        for (int yy = 0; yy < cartCubeDim; yy++)
        {
            for (int xx = 0; xx < cartCubeDim; xx++)
            {
                xval = (oddx > 0) ? xx : cartCubeDim - xx - 1;
                yval = (oddy > 0) ? yy : cartCubeDim - yy - 1;

                k.x = xval - numMissingPts;
                k.y = yval - numMissingPts;
                k.z = zz - numMissingPts;

                kidx = (zz * cartCubeDim + yval) * cartCubeDim + xval;

                dist = k.dot(k);

                if (dist <= m_params.radius2)
                {
                    m_CartesianEncodeData[idx].phaseEncode = k * petraStep;
                    m_CartesianEncodeData[idx].kIndex = kidx;
                    m_CartesianEncodeData[idx].m_distance = sqrt(dist);

                    ++idx;
                }
                else
                {
                    m_CartesianEncodeData[outerIdx].kIndex = ACQINDEX_NOACQ;
                    m_CartesianEncodeData[outerIdx].acqIndex = ACQINDEX_NOACQ;
                    m_CartesianEncodeData[outerIdx].m_distance = sqrt(dist);
                    ++outerIdx;
                }
            }
            oddx *= -1;
        }
        oddy *= -1;
    }
}

void PatternPETRA::calcCartesianPosition(const Loop* seg, const Loop* pe, PETRACartesianData &data)
{
    int32_t numAcqSegs      = seg->acq();
    int32_t numShotPerSeg   = pe->acq();
    int32_t numOuterCond    = seg->dim() - seg->acq();
    int32_t numRadialSeg    = numAcqSegs - m_params.numCartSegs;

    // calculate position of Cartesian segment(s) within set of segments
    int32_t cartSegIndex    = (int)(((m_params.cartSegIndex + 0.5) / 100.0) * (flt64_t)(numAcqSegs - 1));
    if (cartSegIndex > numRadialSeg)
        cartSegIndex = numRadialSeg;
    if (cartSegIndex < 0)
        cartSegIndex = 0;

    data.segStart = cartSegIndex + numOuterCond;
    data.segEnd   = data.segStart + m_params.numCartSegs;

    // calculate position of k=0 (center) within Cartesian segment
    int32_t cartShotCenter  = (int)((m_params.cartShotIndex / 100.0) * (flt64_t)(numShotPerSeg - 1));
    if (cartShotCenter >= numShotPerSeg)
        cartShotCenter = numShotPerSeg - 1;
    if (cartShotCenter < 0)
        cartShotCenter = 0;

    //assume centered within each Cartesian segment
    int32_t cartFront = m_params.numCartEncodes / 2;

    int32_t cartEncodesPerSeg = m_params.numCartEncodes;
    if (m_params.numCartSegs > 0)
    {
        // round down by convention
        // may result in (up to) numCartSegs-1 fewer cartesian encodes acquired
        cartFront /= m_params.numCartSegs;
        cartEncodesPerSeg /= m_params.numCartSegs;
    }
    int32_t cartBack = cartEncodesPerSeg - cartFront;

    // calculate number of shots before beginning Carteisan encoding section
    if (cartShotCenter > cartFront) // add extra shots before Cartesian encodes
    {
        // check for running over end of shot
        if ((cartBack + cartShotCenter) > numShotPerSeg)
        {
            data.shotStart = numShotPerSeg - cartEncodesPerSeg - 1; // push as late as possible
        }
        else
        {
            data.shotStart = cartShotCenter - cartFront; // accurate positioning
        }
    }
    else // push as early as possible
    {
        data.shotStart = 0;
    }

    data.shotEnd = data.shotStart + cartEncodesPerSeg;
}

void PatternPETRA::makeRadialTable(const Loop* seg, const Loop* pe)
{
    int32_t numRadialSeg    = seg->acq() - m_params.numCartSegs;
    int32_t numShotPerSeg   = pe->acq();

    int32_t numRadialAcq = numRadialSeg * numShotPerSeg;
    m_RadialEncodeData.resize(numRadialAcq, EncodeData());

    flt64_t zgrdmod;
    for (int i = 0; i < numRadialAcq; ++i)
    {
        zgrdmod = (flt64_t)(2.0 * i - numRadialAcq + 1) / (flt64_t)numRadialAcq;
        m_RadialEncodeData[i].freqEncode.x = cos(sqrt(M_PI * numRadialAcq) * asin(zgrdmod)) * sqrt(1.0 - zgrdmod * zgrdmod);
        m_RadialEncodeData[i].freqEncode.y = sin(sqrt(M_PI * numRadialAcq) * asin(zgrdmod)) * sqrt(1.0 - zgrdmod * zgrdmod);
        m_RadialEncodeData[i].freqEncode.z = zgrdmod;
    }
}

void PatternPETRA::makeRotationTable()
{
    const Loop *segLp = m_dim->getLoop(0);
    const Loop *peLp = m_dim->getLoop(1);

    // calculate radial table
    makeRadialTable(segLp, peLp);
    int32_t numRadial = static_cast<int32_t>(m_RadialEncodeData.size());

    // calculate Cartesian table
    makeCartesianTable(m_params.numMissingPts);

    // adjust position of Cartesian encodes within set of segments
    PETRACartesianData cart;
    calcCartesianPosition(segLp, peLp, cart);

    int32_t acqCount(0), radCounter(0), cartEncCounter(0);
    LoopIndex seg(segLp);
    LoopIndex pe(peLp);
    LoopSet ls;
    ls.push_back(seg);
    ls.push_back(pe);

    ls.reset();
    for (seg = 0; seg < seg.dim(); seg++)
    {
        for (pe = 0; pe < pe.dim(); pe++)
        {
            if ((seg >= segLp->dummy()) && (pe >= peLp->dummy()))
            {
                // Cartesian section
                if ((cart.segStart <= seg) && (cart.segEnd > seg) && m_EncodeOn)
                {
                    segLp->addAnchor(seg);
                    // Cartesian encoding
                    if ((cartEncCounter < m_params.numCartEncodes) && (cart.shotStart <= pe) && (pe < cart.shotEnd))
                    {
                        (*m_EncodeTable)[ls].freqEncode = m_CartesianEncodeData[cartEncCounter].phaseEncode;
                        (*m_EncodeTable)[ls].kIndex = m_CartesianEncodeData[cartEncCounter].kIndex + numRadial; //position Cartesian data at end of k-space array
                        ++cartEncCounter;
                        peLp->addAnchor(pe);
                    }
                    else if (cart.shotStart > 0 && pe < cart.shotStart) // fill remainder with empty cutout shots
                    {
                        // ramp up gradient slowly
                        (*m_EncodeTable)[ls].freqEncode = m_CartesianEncodeData[cartEncCounter].phaseEncode * ((flt64_t)pe.getIndex() / (flt64_t)cart.shotStart);
                        (*m_EncodeTable)[ls].attrib = VFC_SRTTBL_ATTRB_CUTOUT;
                        (*m_EncodeTable)[ls].kIndex = -1;
                    }
                    else // fill remainder with empty cutout shots
                    {
                        // ramp down gradient slowly
                        (*m_EncodeTable)[ls].freqEncode = m_CartesianEncodeData[cartEncCounter - 1].phaseEncode * ((flt64_t)(pe.dim() - pe.getIndex()) / (flt64_t)(pe.dim() - cart.shotEnd));
                        (*m_EncodeTable)[ls].attrib = VFC_SRTTBL_ATTRB_CUTOUT;
                        (*m_EncodeTable)[ls].kIndex = -1;
                    }
                }
                else
                {
                    (*m_EncodeTable)[ls].freqEncode = m_EncodeOn ? m_RadialEncodeData[radCounter].freqEncode : 0;
                    (*m_EncodeTable)[ls].kIndex = radCounter;
                    ++radCounter;
                }

                (*m_EncodeTable)[ls].acqIndex = acqCount;
                ++acqCount;
            }
            else // conditioning
            {
                (*m_EncodeTable)[ls].freqEncode = 0;
                (*m_EncodeTable)[ls].attrib = VFC_SRTTBL_ATTRB_CONDITION;
                (*m_EncodeTable)[ls].acqIndex = ACQINDEX_NOACQ;
            }
        }
    }

    // add radial anchors
    if (cart.segStart > segLp->dummy())
    {
        // if first segment is radial
        segLp->addAnchor(segLp->dummy());
    }
    else if (cart.segEnd < (int)seg.dim())
    {
        // else, add last segment
        segLp->addAnchor(seg.dim() - 1);
    }
}

void PatternPR::EpvFillPattern()
{
    m_moment->setEncodeAxes(I32Vec3(1, 1, 0));  // 2D (kx & ky)
    m_moment->setPattern(this);

    initTable(m_dim->getLoopList());

    makeRotationTable();

    Table<F64Vec3> phaseEncodeTbl("peenc", F64Vec3());
    createPhaseEncodeTable(phaseEncodeTbl);
    m_moment->setIndexTable(phaseEncodeTbl);

    Table<F64Vec3> freqEncodeTbl("feenc", F64Vec3());
    createFreqEncodeTable(freqEncodeTbl);
    m_moment->setRotationTable(freqEncodeTbl);
}

void PatternPR::generateSortTable(TrajectorySort &sortTable)
{
    sortTable.generateSortTable(getTable());
}

void PatternPR::EpvSaveParams(SeqDBSave& par)
{
    generateSortTable(m_trajectorysort);

    par.xyRatioJet = 1.0f; // FIX - is this correct ?
    m_trajectorysort.saveParams(par.trajectorySort);

    m_angleMgr.saveToStudyDB(par);
}

void PatternPR3D::EpvSaveParams(SeqDBSave& par)
{
    PatternPR::EpvSaveParams(par);
    par.UTESrtTblFlag = m_params.m_newReconFlag ? 1 : 0;
}
