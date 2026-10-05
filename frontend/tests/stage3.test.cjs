// Run: node frontend/tests/stage3.test.cjs (no packages or network required).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../stage3.html'), 'utf8');
const script = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)]
    .map(m => m[1]).find(s => s.includes('stage3Blocked'));
const base = {Age:45, Sex:'Female', BMI:27, fasting_glucose:100, hba1c:6,
    tier:'Moderate', final_risk_level:'Moderate', stage3_eligible:true,
    raw_inputs:{age:45, sex:2, bmi:27, race_ethnicity:3, family_history:0,
        hypertension:0, physical_activity:1, smoking_status:0, fasting_glucose:100, hba1c:6}};
const response = {model_used:'unified_rich_3y', risk_category:'Increased risk',
    next_step:'Discuss your risk factors and a prevention plan with a healthcare professional.',
    time_horizons:{'3_year':{probability:0.042, percentage:4.2}}, warnings:[]};
function harness(changes={}, cacheProfile=null, fetcher=null, cachedResponse=response) {
    const profile={...base, ...changes};
    if (!Object.hasOwn(changes,'raw_inputs')) {
        profile.raw_inputs={...base.raw_inputs};
        for (const [display,raw] of Object.entries({Age:'age',Sex:'sex',BMI:'bmi',fasting_glucose:'fasting_glucose',hba1c:'hba1c'}))
            if(Object.hasOwn(changes,display)) profile.raw_inputs[raw]=changes[display];
    }
    const nodes={}, events={}, calls=[], prints=[];
    const element=()=>({style:{display:'none'},textContent:'',innerHTML:'',listeners:{},
        addEventListener(k,f){this.listeners[k]=f},appendChild(){},scrollIntoView(){}});
    const storage={diabeta_stage2_hybrid_result:JSON.stringify(profile)};
    if(cacheProfile) storage.diabeta_stage3_result=JSON.stringify({...cachedResponse,
        assessmentSnapshot:JSON.stringify(cacheProfile)});
    vm.runInNewContext(script, {console:{debug(){}},window:{print:()=>prints.push(true),location:{hostname:'localhost'},addEventListener:(k,f)=>events[k]=f},
        document:{querySelector:()=>null,getElementById:id=>nodes[id]??=element(),createElement:element},
        localStorage:{getItem:k=>storage[k]??null,setItem:(k,v)=>storage[k]=v,removeItem:k=>delete storage[k]},
        fetch:async (url,options)=>{calls.push(JSON.parse(options.body));return fetcher ? fetcher() : {ok:true,json:async()=>response}}});
    return {nodes, storage, events, calls, prints, download:()=>nodes.downloadStage3ReportBtn.listeners.click(), click:async()=>{
        nodes.generateProjectionsBtn.listeners.click();
        await new Promise(resolve=>setImmediate(resolve));
    }};
}
let passed=0;
async function test(name, fn) {await fn();passed++;console.log('PASS '+name);}
(async()=>{
    await test('report opens print dialog for fresh and restored results',async()=>{
        const fresh=harness();await fresh.click();
        for(const h of [fresh,harness({},base)]) {
            h.download();assert.equal(h.prints.length,1);
            assert.match(h.nodes.reportStatus.textContent,/Save as PDF/);
        }
    });
    await test('report cannot print absent, High or stale results',()=>{
        const stale=harness({},base);
        stale.storage.diabeta_stage2_hybrid_result=JSON.stringify({...base,tier:'High'});
        for(const h of [harness(),harness({tier:'High',final_risk_level:'High'}),stale]) {
            h.download();assert.equal(h.prints.length,0);
            assert.match(h.nodes.reportStatus.textContent,/Generate a Stage 3 result/);
        }
    });
    if (process.env.STAGE2_TRACE) {
        const {payload,resData}=JSON.parse(process.env.STAGE2_TRACE);
        const source=fs.readFileSync(path.join(__dirname,'../stage2.html'),'utf8');
        const start=source.indexOf('                    const tier =',source.indexOf('const resData = await response.json()'));
        const end=source.indexOf('                    try {',start);
        const record=vm.runInNewContext(source.slice(start,end)+';s2Record',{
            payload,resData,hba1cEntered:payload.hba1c!=null,hba1cRaw:payload.hba1c,
            glucoseEntered:payload.fasting_glucose!=null,glucoseRaw:payload.fasting_glucose});
        const h=harness(record);await h.click();
        console.log('TRACE_RESULT:'+JSON.stringify({record,payload:h.calls[0]||null,blocked:h.calls.length===0}));
        return;
    }
    await test('saved request wins over stale display fields',async()=>{
        const h=harness({Age:70,Sex:'Male',BMI:40,hba1c:7,fasting_glucose:150,
            raw_inputs:{...base.raw_inputs}});
        await h.click();assert.deepEqual(h.calls[0],base.raw_inputs);
        assert.equal(h.nodes.stage3ResultsCard.style.display,'block');
    });
    await test('eligible request sends all Stage 2 inputs and displays 3-year result',async()=>{
        const h=harness();await h.click();assert.equal(h.calls.length,1);
        for(const key of Object.keys(base.raw_inputs)) assert.ok(key in h.calls[0],key);
        assert.equal(h.nodes.pct3yr.textContent,'4.20%');
        assert.equal(h.nodes.stage3ResultsCard.style.display,'block');
        assert.equal(h.nodes.riskCategory.textContent,response.risk_category);
        assert.equal(h.nodes.riskNextStep.textContent,response.next_step);
    });
    for (const [probability, category, display] of [
        [0.009999999999999998,'Lower estimated risk','1.00%'],
        [0.01,'Increased risk','1.00%'],
        [0.049999999999999996,'Increased risk','5.00%'],
        [0.05,'Elevated risk','5.00%'],
    ]) await test('unrounded category survives display rounding and cache: '+probability,async()=>{
        const data={...response,risk_category:category,
            time_horizons:{'3_year':{probability,percentage:probability*100}}};
        const h=harness({},null,async()=>({ok:true,json:async()=>data}));
        await h.click();
        assert.equal(h.nodes.pct3yr.textContent,display);
        assert.equal(h.nodes.riskCategory.textContent,category);
        assert.equal(h.nodes.riskPrecision,undefined);
        const cached=JSON.parse(h.storage.diabeta_stage3_result);
        assert.equal(cached.risk_category,category);
        assert.equal(cached.time_horizons['3_year'].probability,probability);
        const restored=harness({},base,null,cached);
        assert.equal(restored.nodes.riskCategory.textContent,category);
        assert.equal(restored.nodes.pct3yr.textContent,display);
    });
    for (const data of JSON.parse(process.env.STAGE3_REAL_RESULTS || '[]')) {
        await test('real model output displays and restores: '+data.risk_category,async()=>{
            const h=harness({},null,async()=>({ok:true,json:async()=>data}));
            await h.click();
            const restored=harness({},base,null,JSON.parse(h.storage.diabeta_stage3_result));
            for (const rendered of [h,restored]) {
                assert.equal(rendered.nodes.riskCategory.textContent,data.risk_category);
                assert.equal(rendered.nodes.riskNextStep.textContent,data.next_step);
                assert.equal(rendered.nodes.pct3yr.textContent,data.risk_percentage.toFixed(2)+'%');
                assert.equal(rendered.nodes.riskPrecision,undefined);
                assert.equal(rendered.nodes.reliabilityBadge.textContent,
                    data.reliability === 'caution' ? 'Use with caution' : 'Research estimate');
                if(data.warnings.length) {
                    assert.equal(rendered.nodes.stage3Warnings.textContent,data.warnings.join(' '));
                    assert.equal(rendered.nodes.stage3Warnings.style.display,'block');
                }
            }
        });
    }
    await test('old rounded-only cache is discarded',()=>{
        const legacy={model_used:'unified_rich_3y',time_horizons:{'3_year':{percentage:1}}};
        const h=harness({},base,null,legacy);
        assert.equal(h.nodes.stage3ResultsCard.style.display,'none');
        assert.ok(!h.storage.diabeta_stage3_result);
    });
    const blockedCases=[['FPG undefined',{fasting_glucose:undefined}],['FPG null',{fasting_glucose:null}],
        ['FPG blank',{fasting_glucose:''}],['FPG whitespace',{fasting_glucose:'  '}],
        ['FPG nonnumeric',{fasting_glucose:'bad'}],['FPG zero',{fasting_glucose:0}],
        ['FPG boolean',{fasting_glucose:true}],['FPG below evidence',{fasting_glucose:50}],
        ['FPG 126',{fasting_glucose:126}],['FPG 140',{fasting_glucose:140}],
        ['HbA1c 6.5',{hba1c:6.5}],['HbA1c 7',{hba1c:7}],
        ['Rule 2 High',{tier:'High',final_risk_level:'High',rule_triggered:'rule_2_model1b_high_probability'}],
        ['Rule 4 High',{tier:'High',final_risk_level:'High',rule_triggered:'rule_4_model1b_low_high_fallback'}],
        ['server ineligible',{stage3_eligible:false}],['missing full assessment',{raw_inputs:null}]];
    for(const [name,changes] of blockedCases) await test(name+' blocks request and cache',async()=>{
        const h=harness(changes,{...base,...changes});await h.click();
        assert.equal(h.calls.length,0);assert.equal(h.nodes.stage3ResultsCard.style.display,'none');
        assert.equal(h.nodes.stage3NotApplicableBanner.style.display,'block');
        assert.ok(!h.storage.diabeta_stage3_result);
    });
    await test('unchanged eligible cache restores',()=>{
        assert.equal(harness({},base).nodes.stage3ResultsCard.style.display,'block');
    });
    await test('legacy cache without assessment identity is discarded',()=>{
        const h=harness({},'legacy');
        assert.equal(h.nodes.stage3ResultsCard.style.display,'none');
        assert.ok(!h.storage.diabeta_stage3_result);
    });
    await test('API rejection clears previously displayed result',async()=>{
        const h=harness({},base,async()=>({ok:false,json:async()=>({error:'Stage 2 High'})}));
        await h.click();assert.equal(h.nodes.stage3ResultsCard.style.display,'none');
        assert.ok(!h.storage.diabeta_stage3_result);
    });
    for(const [key,value] of Object.entries({Age:46,Sex:'Male',BMI:28,fasting_glucose:101,hba1c:6.1,
        tier:'Low',raw_inputs:{...base.raw_inputs,smoking_status:2}})) {
        await test('changed '+key+' invalidates cache',()=>{
            const h=harness({[key]:value},base);
            assert.equal(h.nodes.stage3ResultsCard.style.display,'none');
            assert.ok(!h.storage.diabeta_stage3_result);
        });
    }
    await test('assessment changed in another tab clears visible cache',()=>{
        const h=harness({},base);h.storage.diabeta_stage2_hybrid_result=JSON.stringify({...base,hba1c:7});
        h.events.storage();assert.equal(h.nodes.stage3ResultsCard.style.display,'none');
    });
    await test('assessment changed during request prevents display',async()=>{
        let resolve;const pending=new Promise(r=>resolve=r);const h=harness({},null,()=>pending);
        h.nodes.generateProjectionsBtn.listeners.click();
        h.storage.diabeta_stage2_hybrid_result=JSON.stringify({...base,tier:'High'});
        resolve({ok:true,json:async()=>response});await new Promise(r=>setImmediate(r));
        assert.equal(h.nodes.stage3ResultsCard.style.display,'none');assert.ok(!h.storage.diabeta_stage3_result);
    });
    const stage2=fs.readFileSync(path.join(__dirname,'../stage2.html'),'utf8');
    await test('failed save cannot silently show a new result over old High storage',()=>{
        const save=stage2.indexOf("localStorage.setItem('diabeta_stage2_hybrid_result', JSON.stringify(s2Record))");
        const block=stage2.slice(stage2.lastIndexOf('                    try {',save),
            stage2.indexOf('// Ensure the current assessment',save));
        const previous=JSON.stringify({...base,tier:'High',final_risk_level:'High'});
        const storage={diabeta_stage2_hybrid_result:previous};
        assert.throws(()=>vm.runInNewContext(block,{s2Record:base,payload:base.raw_inputs,
            FORM_STATE_KEY:'draft',traceAssessment(){},localStorage:{
                setItem(){throw new Error('Quota exceeded')},getItem:k=>storage[k]}}),
            /Could not save this assessment/);
        assert.equal(storage.diabeta_stage2_hybrid_result,previous);
    });
    await test('another tab replacing Stage 2 refreshes the visible result',()=>{
        const start=stage2.indexOf("            window.addEventListener('storage'");
        const end=stage2.indexOf('// Mirror every edit',start);
        let handler, refreshed=0;const resultCard={style:{display:'block'}};
        vm.runInNewContext(stage2.slice(start,end),{window:{addEventListener:(key,fn)=>handler=fn},
            resultCard,FORM_STATE_KEY:'diabeta_current_assessment',populateFormFromCurrentAssessment(){refreshed++}});
        handler({key:'diabeta_stage2_hybrid_result'});
        assert.equal(refreshed,1);assert.equal(resultCard.style.display,'none');
    });
    const navigation=stage2.slice(stage2.indexOf("                const continueBtn ="),stage2.indexOf("                resultCard.style.display = 'block';"));
    for(const tier of ['High','Low','Moderate']) await test('Stage 2 navigation '+tier,()=>{
        const nodes={};vm.runInNewContext(navigation,{resData:{tier,stage3_eligible:true},
            document:{getElementById:id=>nodes[id]??={style:{}}}});
        assert.equal(nodes.continueToStage3Btn.style.display,tier==='High'?'none':'');
    });
    await test('public horizon copy and result navigation',()=>{
        assert.doesNotMatch(html,/riskPrecision|Full-precision model estimate/);
        assert.doesNotMatch(html,/Categories use the unrounded estimate|Supported estimate|stabilize HbA1c|GP within 3 months/);
        assert.match(html,/not a clinically validated diagnostic device/);
        assert.match(html,/External clinical validation for Nigerian patients has not been established/);
        const about=fs.readFileSync(path.join(__dirname,'../about.html'),'utf8');
        assert.doesNotMatch(about,/clinical-grade|Privacy guaranteed|data never leaves your device/);
        for(const file of ['index.html','result.html','stage3.html']) {
            const source=fs.readFileSync(path.join(__dirname,'..',file),'utf8');
            assert.doesNotMatch(source,/\b(?:1|2|5|one|two|five)[ -]year\b/i);
            assert.match(source,/approximately (?:3|three)[ -]year/i);
        }
        assert.match(fs.readFileSync(path.join(__dirname,'../result.html'),'utf8'),
            /href="stage2.html"[^>]*id="continueStage3Btn"/);
    });
    console.log(`RESULTS: ${passed} passed, 0 failed`);
})().catch(error=>{console.error(error);process.exitCode=1});
