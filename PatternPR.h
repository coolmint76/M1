//***** CANON MEDICAL SYSTEMS CORPORATION ** COMPANY CONFIDENTIAL *****
///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2007-2013 Canon Medical Systems Corporation, All Rights Reserved
//  Creted: 13-03-21 wheaton
///////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_EncPattPR_H__CB4CDE49_A975_492B_81DB_3C92F4F65BD6__INCLUDED_)
#define AFX_EncPattPR_H__CB4CDE49_A975_492B_81DB_3C92F4F65BD6__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <MR/common/basicdefs.h>
#include <MR/libOSIndepdnt/vector.h>

#include <MR/VfStudy/vfheader.h>
using namespace MRMPlus::common;

#include <iostream>
#include <iomanip>
#include <algorithm>

#include <math.h>
#include <assert.h>
#include <MR/libpattern/PatternBase.h>
#include <MR/libpattern/PatternJET.h>

enum RadialAnglePattern
{
    PR_180 = 0, PR_360 = 1, PR_RECURSIVE = 2, PR_FREE_ANGLE = 3,
    PR_GOLDEN_180 = 4, PR_GOLDEN_360 = 5, PR_GEN_GOLDEN = 6,
};

class AngleMgr
{
    bool        m_active;
    flt64_t     m_cycle;
    flt64_t     m_sample;
    flt64_t     m_cycle_tmplt;
    flt64_t     m_sample_tmplt;

public:
    AngleMgr()
        : m_active(false)
        , m_cycle(0)
        , m_sample(0)
        , m_cycle_tmplt(0)
        , m_sample_tmplt(0)
    {}
    void init()
    {
        m_active = false;
        m_cycle = m_cycle_tmplt = 0;
        m_sample = m_sample_tmplt = 0;
    }
    void setCycle(flt64_t cycle, flt64_t sample)
    {
        m_cycle = m_cycle_tmplt = cycle;
        m_sample = m_sample_tmplt = sample;
    }
    void setCycle(flt64_t cycle, flt64_t sample, flt64_t cycle_tmplt, flt64_t sample_tmplt)
    {
        m_cycle = cycle;
        m_sample = sample;
        m_cycle_tmplt = cycle_tmplt;
        m_sample_tmplt = sample_tmplt;
    }
    void activate() {m_active = true;}
    inline flt64_t getAngle(int index, bool tmplt = false)
    {
        if (tmplt)
            return (flt64_t)index * 2. * M_PI * m_cycle_tmplt / m_sample_tmplt;
        else
            return (flt64_t)index * 2. * M_PI * m_cycle / m_sample;
    }
    void saveToStudyDB(SeqDBSave& par);
};

struct LIBPATTERN_API PatternInputDataPR3D : PatternInputInterface
{
    PatternInputDataPR3D()
    {
        init();
    }

    void init()
    {
        m_encodeType = 0;
        m_newReconFlag = false;
    }

    int32_t             m_encodeType;   //!< e.g. VFC_ENCODE_SEQUENTIAL ...
    bool                m_newReconFlag;
};

struct LIBPATTERN_API PatternInputDataPETRA : PatternInputInterface
{
    PatternInputDataPETRA()
    {
        init();
    }

    void init()
    {
        numMissingPts = 0;
        numCartEncodes = 0;
        numCartSegs = 0;
        cartSegIndex = 0;
        cartShotIndex = 0;
        radius2 = 0;
    }

    int32_t numMissingPts;
    int32_t numCartEncodes;
    int32_t numCartSegs;
    flt64_t cartSegIndex;
    flt64_t cartShotIndex;
    flt64_t radius2;
};

struct LIBPATTERN_API PETRACartesianData
{
    PETRACartesianData()
    {
        init();
    }

    void init()
    {
        segStart = 0;
        segEnd = 0;
        shotStart = 0;
        shotEnd = 0;
    }

    int32_t segStart;
    int32_t segEnd;
    int32_t shotStart;
    int32_t shotEnd;
};

class LIBPATTERN_API PatternPR : public PatternBase<EncodeData>
{
public:
    //! Constructor
    PatternPR(SeqRoot *root, Dimension *d)
        : PatternBase<EncodeData>(root)
        , m_dim(d)
        , m_trajectorysort()
        , m_anglePattern(PR_180)
    {}
    ~PatternPR() {}
    // return empty encode step size
    virtual void EpvInit();
    virtual void setParams(const SeqParam& sp);
    virtual F64Vec3 EpvGetStep() { return F64Vec3(); }
    virtual bool isRadial() const {return true;}
    virtual int32_t getProhibitAFImode() const { return VFC_AFI_RO | VFC_AFI_PE; }
    void EpvFillPattern();
    void EpvSaveParams(SeqDBSave&);

protected:
    virtual void makeRotationTable();
    void generateSortTable(TrajectorySort &sortTable);
    void makeIndexTable(MPlusInt32Vec &indexTable);

    TrajectorySort  m_trajectorysort;
    Dimension*      m_dim;

    RadialAnglePattern  m_anglePattern;
    AngleMgr        m_angleMgr;         //! Angle generator
    SeqParamFlt64   m_cycle;            //! The number of cycles
    SeqParamInt32   m_numSample;        //! Total number of samples for cycles
    SeqParamInt32   m_numSpoke;         //! The number of spokes (PR_RECURSIVE)
    SeqParamInt32   m_numTemplate;      //! The number of template shot
};

class LIBPATTERN_API PatternPR3D : public PatternPR
{
public:
    //! Constructor
    PatternPR3D(SeqRoot *root, Dimension *d)
        : PatternPR(root, d)
    {}
    ~PatternPR3D() {}

    void EpvInit()
    {
        PatternBase::init();
        m_params.init();
    }

    void setParams(const PatternInputInterface& par)
    {
        m_params = static_cast<const PatternInputDataPR3D&>(par);
    }
    void EpvSaveParams(SeqDBSave&);

    PatternInputDataPR3D m_params;

    bool isMixed() const { return true; }

protected:
    virtual void makeRotationTable();
};

class LIBPATTERN_API PatternPETRA : public PatternPR3D
{
public:
    //! Constructor
    PatternPETRA(SeqRoot *root, Dimension *d)
        : PatternPR3D(root, d)
    {}
    ~PatternPETRA()
    {
    }

    void EpvInit()
    {
        PatternBase::init();
        m_params.init();
        m_CartesianEncodeData.clear();
        m_RadialEncodeData.clear();
    }

    void setParams(const PatternInputInterface& par)
    {
        m_params = static_cast<const PatternInputDataPETRA&>(par);
    }

private:
    PatternInputDataPETRA m_params;
    void makeRotationTable();
    void makeCartesianTable(int32_t numMissingPts);
    void makeRadialTable(const Loop* seg, const Loop* pe);
    void calcCartesianPosition(const Loop* seg, const Loop* pe, PETRACartesianData &data);

    vector<EncodeDataMixed> m_CartesianEncodeData;
    vector<EncodeData> m_RadialEncodeData;
};

#endif // !defined(AFX_EncPattPR_H__CB4CDE49_A975_492B_81DB_3C92F4F65BD6__INCLUDED_)
