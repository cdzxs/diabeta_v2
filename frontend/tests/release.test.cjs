// Focused real-handler regressions; synthetic storage only, no browser profile access.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../stage1.html'), 'utf8');
const save = html.match(/function saveCurrentDraft\(\) \{[\s\S]*?(?=            function populateFromCurrentAssessment)/)[0];
const clear = html.match(/function clearTemporaryAssessmentState\(\) \{[\s\S]*?\n        \}/)[0];
const submit = html.slice(html.indexOf('        document\n            .getElementById("assessmentForm")') >= 0
    ? html.indexOf('        document\n            .getElementById("assessmentForm")')
    : html.indexOf('        document\r\n            .getElementById("assessmentForm")'),
    html.indexOf('        /* ============================================================\n           RESTORE CURRENT ASSESSMENT') >= 0
    ? html.indexOf('        /* ============================================================\n           RESTORE CURRENT ASSESSMENT')
    : html.indexOf('        /* ============================================================\r\n           RESTORE CURRENT ASSESSMENT'));
function harness(pending) {
    const values = {patient_name:'Synthetic new patient',age:'45',sex:'0',bmi:'27',race_ethnicity:'3',
        family_history:'0',hypertension:'0',physical_activity:'1',smoking_status:'0'};
    const storage = {diabeta_last_result:'old',diabeta_stage2_hybrid_result:'old',diabeta_stage3_result:'old',
        diabeta_current_assessment:JSON.stringify({hba1c:6,fasting_glucose:110}),diabetaRecords:'[{"id":"history"}]'};
    const nodes = {};
    let handler;
    const form = {addEventListener:(event, fn)=>handler=fn};
    const context = {console,form,window:{location:{}},BACKEND_URL:'http://localhost:5000',
        document:{getElementById:id=>id==='assessmentForm'?form:(nodes[id]??={style:{}})},
        FormData:class {constructor(){this.values={...values}} get(k){return this.values[k]??null}},
        localStorage:{getItem:k=>storage[k]??null,setItem:(k,v)=>storage[k]=v,removeItem:k=>delete storage[k]},
        sessionStorage:{removeItem(){}},
        fetch:async()=>{if(pending) await pending;return {ok:true,json:async()=>({tier:'Low',prob_high:12,person:{'Patient Name':values.patient_name}})}}};
    vm.createContext(context);
    vm.runInContext('let assessmentRevision=0;\n'+save+clear+submit,context);
    return {storage,nodes,context,edit:()=>vm.runInContext('saveCurrentDraft()',context),
        clear:()=>vm.runInContext('clearTemporaryAssessmentState()',context),submit:()=>handler({preventDefault(){},target:form})};
}
(async()=>{
    const stage2=fs.readFileSync(path.join(__dirname,'../stage2.html'),'utf8');
    const populate=stage2.slice(stage2.indexOf('            function populateFormFromCurrentAssessment()'),
        stage2.indexOf('            populateFormFromCurrentAssessment();'));
    for(const sex of [0,'0',2,'2','female',1,'1','male']) {
        for(const mode of ['draft','stage1','stage2']) {
            const fields=Object.fromEntries(['nameInput','addressInput','ageInput','sexSelect','bmiInput','raceSelect',
                'familySelect','hypSelect','paSelect','smokingSelect','hba1cInput','glucoseInput'].map(k=>[k,{value:'old'}]));
            const draft={sex,age:45,bmi:27};
            const stored={draft:JSON.stringify(draft)};
            if(mode==='stage1') stored.diabeta_last_result=JSON.stringify({person:{Age:45,Sex:'Female'}});
            if(mode==='stage2') stored.diabeta_stage2_hybrid_result=JSON.stringify({person:{Age:45},raw_inputs:draft});
            vm.runInNewContext(populate+'\npopulateFormFromCurrentAssessment();',{
                ...fields,fieldRefs:fields,FORM_STATE_KEY:'draft',localStorage:{getItem:k=>stored[k]??null},
                importedPerson:null,isFromStage1:false,loadRaceMap:()=>Promise.resolve(),renderStage2Result(){},labelToRaceCode(){},
                parseLab(){return ''},toggleDirectBtn:{dataset:{bound:true}},importCard:{style:{}},resultCard:{style:{}},importValuesText:{},badges:{}});
            assert.equal(fields.sexSelect.value,[1,'1','male'].includes(sex)?'1':'2');
            assert.equal(fields.nameInput.value,'');
        }
    }
    console.log('PASS Stage 1/Stage 2/draft handoffs map sex codes to actual Stage 2 options and clear old identity');
    let h=harness();h.edit();
    for(const key of ['diabeta_last_result','diabeta_stage2_hybrid_result','diabeta_stage3_result']) assert.equal(h.storage[key],undefined);
    assert.equal(JSON.parse(h.storage.diabeta_current_assessment).hba1c,undefined);
    assert.equal(JSON.parse(h.storage.diabetaRecords)[0].id,'history');
    console.log('PASS Stage 1 edit invalidates downstream results and old labs, preserves history');
    h=harness();await h.submit();
    assert.equal(h.storage.diabeta_stage2_hybrid_result,undefined);
    assert.equal(h.storage.diabeta_stage3_result,undefined);
    assert.equal(JSON.parse(h.storage.diabeta_current_assessment).fasting_glucose,undefined);
    assert.equal(JSON.parse(h.storage.diabeta_last_result).raw_inputs.sex,'0');
    assert.equal(JSON.parse(h.storage.diabetaRecords).length,2);
    console.log('PASS fresh Stage 1 completion replaces current snapshot and saves once');
    for(const action of ['edit','clear']) {
        let release;const pending=new Promise(r=>release=r);h=harness(pending);
        const request=h.submit();h[action]();release();await request;
        assert.equal(h.storage.diabeta_last_result,undefined);
        assert.match(h.nodes.formError.textContent,/inputs changed/);
        assert.equal(JSON.parse(h.storage.diabetaRecords).length,1);
        console.log('PASS in-flight Stage 1 '+action+' prevents stale result and record');
    }
    const resultHtml=fs.readFileSync(path.join(__dirname,'../result.html'),'utf8');
    const script=[...resultHtml.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map(x=>x[1]).find(x=>x.includes('diabeta_last_result'));
    for(const stored of [null,'broken',JSON.stringify({fromDirectStage2:true,person:{}})]) {
        const window={location:{}};
        vm.runInNewContext(script,{window,localStorage:{getItem:()=>stored}});
        assert.equal(window.location.href,stored?.includes('fromDirectStage2')?'stage2.html':'stage1.html');
    }
    console.log('PASS missing/corrupt/direct-entry snapshots cannot display a Stage 1 score');
})().catch(error=>{console.error(error);process.exitCode=1});
